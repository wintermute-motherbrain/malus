use std::cell::Cell;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::AnyThread;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice,
    MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dtype {
    F32, F16, Bf16,
    I8, I16, I32, I64,
    U8, U16, U32, U64,
}

impl Dtype {
    pub fn from_tag(tag: i32) -> Self {
        match tag {
            0  => Dtype::F32,
            1  => Dtype::F16,
            2  => Dtype::Bf16,
            3  => Dtype::I8,
            4  => Dtype::I16,
            5  => Dtype::I32,
            6  => Dtype::I64,
            7  => Dtype::U8,
            8  => Dtype::U16,
            9  => Dtype::U32,
            10 => Dtype::U64,
            _  => panic!("malus: unknown dtype tag {tag}"),
        }
    }

    pub fn to_tag(&self) -> i32 {
        match self {
            Dtype::F32 => 0,  Dtype::F16 => 1,  Dtype::Bf16 => 2,
            Dtype::I8 => 3,   Dtype::I16 => 4,  Dtype::I32 => 5,
            Dtype::I64 => 6,  Dtype::U8 => 7,   Dtype::U16 => 8,
            Dtype::U32 => 9,  Dtype::U64 => 10,
        }
    }

    pub fn element_size(&self) -> usize {
        match self {
            Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
            Dtype::F16 | Dtype::Bf16 | Dtype::I16 | Dtype::U16 => 2,
            Dtype::I8 | Dtype::U8 => 1,
            Dtype::I64 | Dtype::U64 => 8,
        }
    }
}

struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    current_command_buffer: Mutex<Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>>,
    pipelines: Mutex<HashMap<u64, Retained<ProtocolObject<dyn MTLComputePipelineState>>>>,
}

// Safety: Metal objects are thread-safe per the Metal specification.
// Access to the command buffer is already serialized through the Mutex.
unsafe impl Send for MetalContext {}
unsafe impl Sync for MetalContext {}

static CONTEXT: OnceLock<MetalContext> = OnceLock::new();

fn context() -> &'static MetalContext {
    CONTEXT.get_or_init(|| {
        let device = MTLCreateSystemDefaultDevice()
            .expect("malus: no Metal device available");
        let command_queue = device.newCommandQueue()
            .expect("malus: failed to create MTLCommandQueue");
        MetalContext {
            device,
            command_queue,
            current_command_buffer: Mutex::new(None),
            pipelines: Mutex::new(HashMap::new()),
        }
    })
}

pub struct TensorBuffer {
    pub buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub dtype: Dtype,
    pub len: usize,
    pub shape: Vec<usize>,
    /// Reference count for M10 RC paths. Initialized to 1 at allocation;
    /// freed when decremented to 0 via `tensor_release`.  `tensor_free`
    /// delegates to `tensor_release`, so all free paths share one code path.
    pub ref_count: std::sync::atomic::AtomicUsize,
}

// ── Runtime init: compile all MSL kernels ─────────────────────────────────────

pub fn runtime_init(registry: &HashMap<u64, String>) {
    let ctx = context();
    let mut pipelines = ctx.pipelines.lock().unwrap();

    for (id, source) in registry {
        let source_ns = NSString::from_str(source);
        let library = ctx.device
            .newLibraryWithSource_options_error(&source_ns, None)
            .unwrap_or_else(|e| panic!("malus: MSL compilation failed for kernel {}: {}", id, e));
        let func_name = format!("malus_kernel_{}", id);
        let func_name_ns = NSString::from_str(&func_name);
        let function = library
            .newFunctionWithName(&func_name_ns)
            .unwrap_or_else(|| panic!("malus: kernel function '{}' not found", func_name));
        let pipeline = ctx.device
            .newComputePipelineStateWithFunction_error(&*function)
            .unwrap_or_else(|e| panic!("malus: pipeline creation failed for kernel {}: {}", id, e));
        pipelines.insert(*id, pipeline);
    }
}

// ── Tensor allocation ─────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn tensor_alloc_gpu(
    dtype: i32,
    shape_ptr: *const usize,
    ndims: usize,
    data: *const f32,
) -> i64 {
    let dt = Dtype::from_tag(dtype);
    match dt {
        Dtype::F32 | Dtype::I32 | Dtype::I64 => {}
        _ => panic!("malus: unsupported dtype {:?} (tag {}); supported: f32, i32, i64", dt, dtype),
    }
    let shape = unsafe { std::slice::from_raw_parts(shape_ptr, ndims).to_vec() };
    let n: usize = shape.iter().product();
    let byte_len = n * dt.element_size();

    let ctx = context();
    // Metal rejects a 0-byte allocation; use a 1-byte placeholder so zero-length
    // tensors (`zeros(0)`, empty kernel output) are safe to allocate and free.
    // `tb.len` stays = n (0) so slices and shape queries remain correct.
    let alloc_len = byte_len.max(1);
    let buffer = ctx.device
        .newBufferWithLength_options(alloc_len, MTLResourceOptions::StorageModeShared)
        .expect("malus: failed to allocate MTLBuffer");

    if !data.is_null() && n > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(
                data as *const u8,
                buffer.contents().as_ptr() as *mut u8,
                byte_len,
            );
        }
    }

    let tb = Box::new(TensorBuffer {
        buffer,
        dtype: dt,
        len: n,
        shape,
        ref_count: std::sync::atomic::AtomicUsize::new(1),
    });
    Box::into_raw(tb) as i64
}

#[no_mangle]
pub extern "C" fn tensor_alloc_zeros_gpu(shape_ptr: *const usize, ndims: usize) -> i64 {
    // Metal allocates zero-initialized StorageModeShared buffers by default.
    tensor_alloc_gpu(0, shape_ptr, ndims, std::ptr::null())
}

#[no_mangle]
pub extern "C" fn tensor_alloc_ones_gpu(shape_ptr: *const usize, ndims: usize) -> i64 {
    let shape = unsafe { std::slice::from_raw_parts(shape_ptr, ndims) };
    let n: usize = shape.iter().product();
    let ones_data: Vec<f32> = vec![1.0f32; n];
    tensor_alloc_gpu(0, shape_ptr, ndims, ones_data.as_ptr())
}

/// Increment the reference count of the tensor at `handle`.
///
/// M9 never calls this; it exists so M10 struct-field RC paths have the ABI
/// available without requiring a runtime version bump.
#[no_mangle]
pub extern "C" fn tensor_retain(handle: i64) {
    if handle == 0 {
        return;
    }
    let tb = unsafe { &*(handle as *const TensorBuffer) };
    tb.ref_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Decrement the reference count.  Frees the tensor when it reaches zero.
///
/// All free paths (including `tensor_free`) go through here so the ownership
/// invariant is single-sourced.
#[no_mangle]
pub extern "C" fn tensor_release(handle: i64) {
    if handle == 0 {
        return;
    }
    let tb = unsafe { &*(handle as *const TensorBuffer) };
    // AcqRel: Acquire on the last decrement so all prior writes to the buffer
    // are visible before the drop; Release on all earlier decrements.
    if tb.ref_count.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) == 1 {
        crate::tape::tape_on_release(handle);
        unsafe { drop(Box::from_raw(handle as *mut TensorBuffer)); }
    }
}

/// Free a tensor unconditionally.  Delegates to `tensor_release` so the
/// decrement-to-zero path is shared.  Callers must not use `handle` after this.
#[no_mangle]
pub extern "C" fn tensor_free(handle: i64) {
    tensor_release(handle);
}

#[no_mangle]
pub extern "C" fn tensor_print(handle: i64) {
    if handle == 0 {
        print!("[]");
        return;
    }
    let tb = unsafe { &*(handle as *const TensorBuffer) };
    match tb.dtype {
        Dtype::I32 => {
            let slice = unsafe {
                std::slice::from_raw_parts(tb.buffer.contents().as_ptr() as *const i32, tb.len)
            };
            print_nd_i32(slice, &tb.shape, 0);
        }
        Dtype::I64 => {
            let slice = unsafe {
                std::slice::from_raw_parts(tb.buffer.contents().as_ptr() as *const i64, tb.len)
            };
            print_nd_i64(slice, &tb.shape, 0);
        }
        _ => {
            let slice = unsafe {
                std::slice::from_raw_parts(tb.buffer.contents().as_ptr() as *const f32, tb.len)
            };
            print_nd(slice, &tb.shape, 0);
        }
    }
}

fn print_nd(data: &[f32], shape: &[usize], offset: usize) {
    if shape.is_empty() {
        print!("{}", data[offset]);
        return;
    }
    if shape.len() == 1 {
        print!("[");
        for i in 0..shape[0] {
            if i > 0 { print!(", "); }
            print!("{}", data[offset + i]);
        }
        print!("]");
        return;
    }
    let stride: usize = shape[1..].iter().product();
    print!("[");
    for i in 0..shape[0] {
        if i > 0 { print!(", "); }
        print_nd(data, &shape[1..], offset + i * stride);
    }
    print!("]");
}

fn print_nd_i32(data: &[i32], shape: &[usize], offset: usize) {
    if shape.len() == 1 {
        print!("[");
        for i in 0..shape[0] {
            if i > 0 { print!(", "); }
            print!("{}", data[offset + i]);
        }
        print!("]");
        return;
    }
    let stride: usize = shape[1..].iter().product();
    print!("[");
    for i in 0..shape[0] {
        if i > 0 { print!(", "); }
        print_nd_i32(data, &shape[1..], offset + i * stride);
    }
    print!("]");
}

fn print_nd_i64(data: &[i64], shape: &[usize], offset: usize) {
    if shape.len() == 1 {
        print!("[");
        for i in 0..shape[0] {
            if i > 0 { print!(", "); }
            print!("{}", data[offset + i]);
        }
        print!("]");
        return;
    }
    let stride: usize = shape[1..].iter().product();
    print!("[");
    for i in 0..shape[0] {
        if i > 0 { print!(", "); }
        print_nd_i64(data, &shape[1..], offset + i * stride);
    }
    print!("]");
}

#[no_mangle]
pub extern "C" fn tensor_len(handle: i64) -> i64 {
    let tb = unsafe { &*(handle as *const TensorBuffer) };
    tb.len as i64
}

// ── GPU barrier ───────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn gpu_barrier() {
    let ctx = context();
    let mut guard = ctx.current_command_buffer.lock().unwrap();
    if let Some(cmd_buf) = guard.take() {
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
    }
}

// ── Eager CPU ops (matmul, transpose, sum) ────────────────────────────────────
//
// CPU reference matmul — kept as ground-truth for MPS correctness tests.
pub fn tensor_matmul_cpu(handle_a: i64, handle_b: i64) -> i64 {
    gpu_barrier();
    let a = unsafe { &*(handle_a as *const TensorBuffer) };
    let b = unsafe { &*(handle_b as *const TensorBuffer) };

    match (a.shape.len(), b.shape.len()) {
        (2, 2) => {
            let (m, k) = (a.shape[0], a.shape[1]);
            let (k2, n) = (b.shape[0], b.shape[1]);
            if k != k2 {
                panic!(
                    "malus: matmul shape mismatch: [{m}x{k}] @ [{k2}x{n}] — inner dims {k} != {k2}\n  \
                     left shape:  {:?}\n  right shape: {:?}",
                    a.shape, b.shape
                );
            }
            let a_data = unsafe { std::slice::from_raw_parts(a.buffer.contents().as_ptr() as *const f32, a.len) };
            let b_data = unsafe { std::slice::from_raw_parts(b.buffer.contents().as_ptr() as *const f32, b.len) };
            let mut out_data = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    for kk in 0..k {
                        out_data[i * n + j] += a_data[i * k + kk] * b_data[kk * n + j];
                    }
                }
            }
            let out_shape = [m, n];
            tensor_alloc_gpu(0, out_shape.as_ptr(), 2, out_data.as_ptr())
        }
        (3, 3) => {
            let (batch, m, k) = (a.shape[0], a.shape[1], a.shape[2]);
            let (batch2, k2, n) = (b.shape[0], b.shape[1], b.shape[2]);
            if batch != batch2 {
                panic!(
                    "malus: batched matmul batch dims must match: {} vs {}\n  \
                     left shape:  {:?}\n  right shape: {:?}",
                    batch, batch2, a.shape, b.shape
                );
            }
            if k != k2 {
                panic!(
                    "malus: batched matmul inner dims must match: {} vs {}\n  \
                     left shape:  {:?}\n  right shape: {:?}",
                    k, k2, a.shape, b.shape
                );
            }
            let a_data = unsafe { std::slice::from_raw_parts(a.buffer.contents().as_ptr() as *const f32, a.len) };
            let b_data = unsafe { std::slice::from_raw_parts(b.buffer.contents().as_ptr() as *const f32, b.len) };
            let out_len = batch * m * n;
            let mut out_data = vec![0.0f32; out_len];
            for bx in 0..batch {
                let a_off = bx * m * k;
                let b_off = bx * k * n;
                let c_off = bx * m * n;
                for i in 0..m {
                    for j in 0..n {
                        for kk in 0..k {
                            out_data[c_off + i * n + j] +=
                                a_data[a_off + i * k + kk] * b_data[b_off + kk * n + j];
                        }
                    }
                }
            }
            let out_shape = [batch, m, n];
            tensor_alloc_gpu(0, out_shape.as_ptr(), 3, out_data.as_ptr())
        }
        (3, 2) => {
            let (batch, m, k) = (a.shape[0], a.shape[1], a.shape[2]);
            let (k2, n) = (b.shape[0], b.shape[1]);
            if k != k2 {
                panic!(
                    "malus: matmul shape mismatch: [{batch}x{m}x{k}] @ [{k2}x{n}] — inner dims {k} != {k2}\n  \
                     left shape:  {:?}\n  right shape: {:?}",
                    a.shape, b.shape
                );
            }
            let a_data = unsafe { std::slice::from_raw_parts(a.buffer.contents().as_ptr() as *const f32, a.len) };
            let b_data = unsafe { std::slice::from_raw_parts(b.buffer.contents().as_ptr() as *const f32, b.len) };
            let out_len = batch * m * n;
            let mut out_data = vec![0.0f32; out_len];
            for bx in 0..batch {
                let a_off = bx * m * k;
                let c_off = bx * m * n;
                for i in 0..m {
                    for j in 0..n {
                        for kk in 0..k {
                            out_data[c_off + i * n + j] += a_data[a_off + i * k + kk] * b_data[kk * n + j];
                        }
                    }
                }
            }
            let out_shape = [batch, m, n];
            tensor_alloc_gpu(0, out_shape.as_ptr(), 3, out_data.as_ptr())
        }
        _ => panic!(
            "malus: tensor_matmul requires both inputs to be 2-D, both 3-D, or (3-D, 2-D)\n  \
             left shape:  {:?}\n  right shape: {:?}",
            a.shape, b.shape
        ),
    }
}

// MPS matmul — dispatches to the AMX coprocessor via MPSMatrixMultiplication.
// Returns a ready tensor (gpu_barrier() + commit + waitUntilCompleted inside).
#[no_mangle]
pub extern "C" fn tensor_matmul(handle_a: i64, handle_b: i64) -> i64 {
    gpu_barrier(); // flush any pending element-wise kernels before MPS reads buffers
    let a = unsafe { &*(handle_a as *const TensorBuffer) };
    let b = unsafe { &*(handle_b as *const TensorBuffer) };
    let ctx = context();

    match (a.shape.len(), b.shape.len()) {
        (2, 2) => {
            let (m, k) = (a.shape[0], a.shape[1]);
            let (k2, n) = (b.shape[0], b.shape[1]);
            if k != k2 {
                panic!(
                    "malus: matmul shape mismatch: [{m}x{k}] @ [{k2}x{n}] — inner dims {k} != {k2}\n  \
                     left shape:  {:?}\n  right shape: {:?}",
                    a.shape, b.shape
                );
            }
            if m == 0 || k == 0 || n == 0 {
                return tensor_matmul_cpu(handle_a, handle_b);
            }
            let out_shape = [m, n];
            let c_handle = tensor_alloc_gpu(0, out_shape.as_ptr(), 2, std::ptr::null());
            let c_tb = unsafe { &*(c_handle as *const TensorBuffer) };
            unsafe {
                let desc_a = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(m, k, k * 4, MPSDataType::Float32);
                let desc_b = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(k, n, n * 4, MPSDataType::Float32);
                let desc_c = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(m, n, n * 4, MPSDataType::Float32);
                let mat_a = MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(),&*a.buffer, &desc_a);
                let mat_b = MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(),&*b.buffer, &desc_b);
                let mat_c = MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(),&*c_tb.buffer, &desc_c);
                let mm = MPSMatrixMultiplication::initWithDevice_resultRows_resultColumns_interiorColumns(
                    MPSMatrixMultiplication::alloc(),&*ctx.device, m, n, k);
                let cmd_buf = ctx.command_queue.commandBuffer()
                    .expect("malus: MPS matmul failed to create command buffer");
                mm.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(&*cmd_buf, &mat_a, &mat_b, &mat_c);
                cmd_buf.commit();
                cmd_buf.waitUntilCompleted();
            }
            c_handle
        }
        (3, 3) => {
            let (batch, m, k) = (a.shape[0], a.shape[1], a.shape[2]);
            let (batch2, k2, n) = (b.shape[0], b.shape[1], b.shape[2]);
            if batch != batch2 {
                panic!(
                    "malus: batched matmul batch dims must match: {} vs {}\n  \
                     left shape:  {:?}\n  right shape: {:?}",
                    batch, batch2, a.shape, b.shape
                );
            }
            if k != k2 {
                panic!(
                    "malus: batched matmul inner dims must match: {} vs {}\n  \
                     left shape:  {:?}\n  right shape: {:?}",
                    k, k2, a.shape, b.shape
                );
            }
            if batch == 0 || m == 0 || k == 0 || n == 0 {
                return tensor_matmul_cpu(handle_a, handle_b);
            }
            let out_shape = [batch, m, n];
            let c_handle = tensor_alloc_gpu(0, out_shape.as_ptr(), 3, std::ptr::null());
            let c_tb = unsafe { &*(c_handle as *const TensorBuffer) };
            unsafe {
                let desc_a = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(m, k, k * 4, MPSDataType::Float32);
                let desc_b = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(k, n, n * 4, MPSDataType::Float32);
                let desc_c = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(m, n, n * 4, MPSDataType::Float32);
                let mm = MPSMatrixMultiplication::initWithDevice_resultRows_resultColumns_interiorColumns(
                    MPSMatrixMultiplication::alloc(),&*ctx.device, m, n, k);
                let cmd_buf = ctx.command_queue.commandBuffer()
                    .expect("malus: MPS batched matmul failed to create command buffer");
                for bx in 0..batch {
                    let a_off = bx * m * k * 4;
                    let b_off = bx * k * n * 4;
                    let c_off = bx * m * n * 4;
                    let mat_a = MPSMatrix::initWithBuffer_offset_descriptor(MPSMatrix::alloc(),&*a.buffer, a_off, &desc_a);
                    let mat_b = MPSMatrix::initWithBuffer_offset_descriptor(MPSMatrix::alloc(),&*b.buffer, b_off, &desc_b);
                    let mat_c = MPSMatrix::initWithBuffer_offset_descriptor(MPSMatrix::alloc(),&*c_tb.buffer, c_off, &desc_c);
                    mm.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(&*cmd_buf, &mat_a, &mat_b, &mat_c);
                }
                cmd_buf.commit();
                cmd_buf.waitUntilCompleted();
            }
            c_handle
        }
        (3, 2) => {
            // Broadcast 2-D weight over the batch dim: (B, M, K) @ (K, N) → (B, M, N)
            let (batch, m, k) = (a.shape[0], a.shape[1], a.shape[2]);
            let (k2, n) = (b.shape[0], b.shape[1]);
            if k != k2 {
                panic!(
                    "malus: matmul shape mismatch: [{batch}x{m}x{k}] @ [{k2}x{n}] — inner dims {k} != {k2}\n  \
                     left shape:  {:?}\n  right shape: {:?}",
                    a.shape, b.shape
                );
            }
            if batch == 0 || m == 0 || k == 0 || n == 0 {
                return tensor_matmul_cpu(handle_a, handle_b);
            }
            let out_shape = [batch, m, n];
            let c_handle = tensor_alloc_gpu(0, out_shape.as_ptr(), 3, std::ptr::null());
            let c_tb = unsafe { &*(c_handle as *const TensorBuffer) };
            unsafe {
                let desc_a = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(m, k, k * 4, MPSDataType::Float32);
                let desc_b = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(k, n, n * 4, MPSDataType::Float32);
                let desc_c = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(m, n, n * 4, MPSDataType::Float32);
                // B is the same 2-D matrix for every batch slice
                let mat_b = MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(),&*b.buffer, &desc_b);
                let mm = MPSMatrixMultiplication::initWithDevice_resultRows_resultColumns_interiorColumns(
                    MPSMatrixMultiplication::alloc(),&*ctx.device, m, n, k);
                let cmd_buf = ctx.command_queue.commandBuffer()
                    .expect("malus: MPS 3x2 matmul failed to create command buffer");
                for bx in 0..batch {
                    let a_off = bx * m * k * 4;
                    let c_off = bx * m * n * 4;
                    let mat_a = MPSMatrix::initWithBuffer_offset_descriptor(MPSMatrix::alloc(),&*a.buffer, a_off, &desc_a);
                    let mat_c = MPSMatrix::initWithBuffer_offset_descriptor(MPSMatrix::alloc(),&*c_tb.buffer, c_off, &desc_c);
                    mm.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(&*cmd_buf, &mat_a, &mat_b, &mat_c);
                }
                cmd_buf.commit();
                cmd_buf.waitUntilCompleted();
            }
            c_handle
        }
        _ => panic!(
            "malus: tensor_matmul requires both inputs to be 2-D, both 3-D, or (3-D, 2-D)\n  \
             left shape:  {:?}\n  right shape: {:?}",
            a.shape, b.shape
        ),
    }
}

#[no_mangle]
pub extern "C" fn tensor_transpose(handle: i64) -> i64 {
    gpu_barrier();
    let tb = unsafe { &*(handle as *const TensorBuffer) };

    if tb.shape.len() != 2 {
        panic!("malus: tensor_transpose requires a 2-D tensor, got shape {:?}", tb.shape);
    }
    let m = tb.shape[0];
    let n = tb.shape[1];

    let in_data = unsafe { std::slice::from_raw_parts(tb.buffer.contents().as_ptr() as *const f32, tb.len) };
    let mut out_data = vec![0.0f32; tb.len];
    for i in 0..m {
        for j in 0..n {
            out_data[j * m + i] = in_data[i * n + j];
        }
    }

    let out_shape = [n, m];
    tensor_alloc_gpu(0, out_shape.as_ptr(), 2, out_data.as_ptr())
}

#[no_mangle]
pub extern "C" fn tensor_sum(handle: i64) -> i64 {
    gpu_barrier();
    let tb = unsafe { &*(handle as *const TensorBuffer) };
    let data = unsafe { std::slice::from_raw_parts(tb.buffer.contents().as_ptr() as *const f32, tb.len) };
    let total: f32 = data.iter().sum();
    let shape = [1usize];
    tensor_alloc_gpu(0, shape.as_ptr(), 1, &total)
}

// ── Broadcasting + axis reductions ───────────────────────────────────────────
//
// Broadcasting: NumPy right-aligned rule (D1/D2 in M16 plan). Shapes are
// runtime-only (ADR-0013); detection and validation happen here, not in sema.
//
// Equal-shape element-wise ops keep the existing GPU kernel path; broadcasting
// falls back to an eager CPU loop and returns a ready tensor.

fn broadcast_result_shape(sa: &[usize], sb: &[usize]) -> Vec<usize> {
    let n = sa.len().max(sb.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let da = if i + sa.len() >= n { sa[i + sa.len() - n] } else { 1 };
        let db = if i + sb.len() >= n { sb[i + sb.len() - n] } else { 1 };
        let d = match (da, db) {
            (x, y) if x == y => x,
            (x, 1) => x,
            (1, y) => y,
            _ => panic!(
                "malus: broadcast shape mismatch at dim {}: {} vs {}\n  left shape:  {:?}\n  right shape: {:?}",
                i, da, db, sa, sb
            ),
        };
        out.push(d);
    }
    out
}

fn broadcast_cpu_loop(
    a_data: &[f32], a_shape: &[usize],
    b_data: &[f32], b_shape: &[usize],
    out_data: &mut [f32], out_shape: &[usize],
    op: impl Fn(f32, f32) -> f32,
) {
    let n = out_shape.len();
    let mut pa = vec![1usize; n];
    let mut pb = vec![1usize; n];
    let offset_a = n - a_shape.len();
    let offset_b = n - b_shape.len();
    for (i, &d) in a_shape.iter().enumerate() { pa[offset_a + i] = d; }
    for (i, &d) in b_shape.iter().enumerate() { pb[offset_b + i] = d; }
    for flat in 0..out_data.len() {
        let mut rem = flat;
        let mut out_idx = vec![0usize; n];
        for dim in (0..n).rev() {
            out_idx[dim] = rem % out_shape[dim];
            rem /= out_shape[dim];
        }
        let mut a_flat = 0usize;
        let mut b_flat = 0usize;
        for dim in 0..n {
            a_flat = a_flat * pa[dim] + (out_idx[dim] % pa[dim]);
            b_flat = b_flat * pb[dim] + (out_idx[dim] % pb[dim]);
        }
        out_data[flat] = op(a_data[a_flat], b_data[b_flat]);
    }
}

fn tensor_broadcast_op(kernel_id: u64, a_h: i64, b_h: i64, op: impl Fn(f32, f32) -> f32) -> i64 {
    let a = unsafe { &*(a_h as *const TensorBuffer) };
    let b = unsafe { &*(b_h as *const TensorBuffer) };
    if a.shape == b.shape {
        let handles = [a_h, b_h];
        kernel_dispatch(kernel_id, handles.as_ptr(), 2)
    } else {
        gpu_barrier();
        let out_shape = broadcast_result_shape(&a.shape, &b.shape);
        let a_data = unsafe { std::slice::from_raw_parts(a.buffer.contents().as_ptr() as *const f32, a.len) };
        let b_data = unsafe { std::slice::from_raw_parts(b.buffer.contents().as_ptr() as *const f32, b.len) };
        let out_len: usize = out_shape.iter().product();
        let mut out_data = vec![0.0f32; out_len.max(1)];
        broadcast_cpu_loop(a_data, &a.shape, b_data, &b.shape, &mut out_data, &out_shape, op);
        tensor_alloc_gpu(0, out_shape.as_ptr(), out_shape.len(), out_data.as_ptr())
    }
}

#[no_mangle]
pub extern "C" fn tensor_broadcast_add(kernel_id: u64, a: i64, b: i64) -> i64 {
    tensor_broadcast_op(kernel_id, a, b, |x, y| x + y)
}

#[no_mangle]
pub extern "C" fn tensor_broadcast_sub(kernel_id: u64, a: i64, b: i64) -> i64 {
    tensor_broadcast_op(kernel_id, a, b, |x, y| x - y)
}

#[no_mangle]
pub extern "C" fn tensor_broadcast_mul(kernel_id: u64, a: i64, b: i64) -> i64 {
    tensor_broadcast_op(kernel_id, a, b, |x, y| x * y)
}

#[no_mangle]
pub extern "C" fn tensor_broadcast_div(kernel_id: u64, a: i64, b: i64) -> i64 {
    tensor_broadcast_op(kernel_id, a, b, |x, y| x / y)
}

// ── Axis reduction helpers ────────────────────────────────────────────────────

fn normalize_axis(axis: i32, ndims: usize) -> usize {
    let a = if axis < 0 { axis + ndims as i32 } else { axis };
    if a < 0 || (a as usize) >= ndims {
        panic!("malus: axis {} is out of range for tensor with {} dimensions", axis, ndims);
    }
    a as usize
}

fn reduce_axis_shape(in_shape: &[usize], axis: usize, keepdim: bool) -> Vec<usize> {
    if keepdim {
        let mut s = in_shape.to_vec();
        s[axis] = 1;
        s
    } else {
        in_shape.iter().enumerate()
            .filter(|&(i, _)| i != axis)
            .map(|(_, &d)| d)
            .collect()
    }
}

fn reduce_out_flat(in_idx: &[usize], axis: usize, keepdim: bool, out_shape: &[usize]) -> usize {
    let mut flat = 0usize;
    let mut out_i = 0usize;
    for (dim, &idx) in in_idx.iter().enumerate() {
        if dim == axis {
            if keepdim {
                flat = flat * out_shape[out_i]; // out_idx = 0 for reduced dim
                out_i += 1;
            }
        } else {
            flat = flat * out_shape[out_i] + idx;
            out_i += 1;
        }
    }
    flat
}

fn reduce_elements(
    in_data: &[f32],
    in_shape: &[usize],
    axis: usize,
    keepdim: bool,
    out_shape: &[usize],
    out_data: &mut [f32],
    reduce_fn: impl Fn(f32, f32) -> f32,
) {
    let ndims = in_shape.len();
    for flat in 0..in_data.len() {
        let mut rem = flat;
        let mut in_idx = vec![0usize; ndims];
        for dim in (0..ndims).rev() {
            in_idx[dim] = rem % in_shape[dim];
            rem /= in_shape[dim];
        }
        let out_flat = reduce_out_flat(&in_idx, axis, keepdim, out_shape);
        out_data[out_flat] = reduce_fn(out_data[out_flat], in_data[flat]);
    }
}

#[no_mangle]
pub extern "C" fn tensor_reduce_sum_axis(h: i64, axis: i64, keepdim: i64) -> i64 {
    gpu_barrier();
    let tb = unsafe { &*(h as *const TensorBuffer) };
    let axis_u = normalize_axis(axis as i32, tb.shape.len());
    let keepdim_b = keepdim != 0;
    let out_shape = reduce_axis_shape(&tb.shape, axis_u, keepdim_b);
    let out_len: usize = out_shape.iter().product::<usize>().max(1);
    let mut out_data = vec![0.0f32; out_len];
    let in_data = unsafe { std::slice::from_raw_parts(tb.buffer.contents().as_ptr() as *const f32, tb.len) };
    reduce_elements(in_data, &tb.shape, axis_u, keepdim_b, &out_shape, &mut out_data, |a, b| a + b);
    tensor_alloc_gpu(0, out_shape.as_ptr(), out_shape.len(), out_data.as_ptr())
}

#[no_mangle]
pub extern "C" fn tensor_reduce_mean_axis(h: i64, axis: i64, keepdim: i64) -> i64 {
    gpu_barrier();
    let tb = unsafe { &*(h as *const TensorBuffer) };
    let axis_u = normalize_axis(axis as i32, tb.shape.len());
    let keepdim_b = keepdim != 0;
    let n = tb.shape[axis_u] as f32;
    let out_shape = reduce_axis_shape(&tb.shape, axis_u, keepdim_b);
    let out_len: usize = out_shape.iter().product::<usize>().max(1);
    let mut out_data = vec![0.0f32; out_len];
    let in_data = unsafe { std::slice::from_raw_parts(tb.buffer.contents().as_ptr() as *const f32, tb.len) };
    reduce_elements(in_data, &tb.shape, axis_u, keepdim_b, &out_shape, &mut out_data, |a, b| a + b);
    for v in out_data.iter_mut() { *v /= n; }
    tensor_alloc_gpu(0, out_shape.as_ptr(), out_shape.len(), out_data.as_ptr())
}

#[no_mangle]
pub extern "C" fn tensor_reduce_max_axis(h: i64, axis: i64, keepdim: i64) -> i64 {
    gpu_barrier();
    let tb = unsafe { &*(h as *const TensorBuffer) };
    let axis_u = normalize_axis(axis as i32, tb.shape.len());
    let keepdim_b = keepdim != 0;
    let out_shape = reduce_axis_shape(&tb.shape, axis_u, keepdim_b);
    let out_len: usize = out_shape.iter().product::<usize>().max(1);
    let mut out_data = vec![f32::NEG_INFINITY; out_len];
    let in_data = unsafe { std::slice::from_raw_parts(tb.buffer.contents().as_ptr() as *const f32, tb.len) };
    reduce_elements(in_data, &tb.shape, axis_u, keepdim_b, &out_shape, &mut out_data, f32::max);
    tensor_alloc_gpu(0, out_shape.as_ptr(), out_shape.len(), out_data.as_ptr())
}

#[no_mangle]
pub extern "C" fn tensor_reduce_var_axis(h: i64, axis: i64, keepdim: i64) -> i64 {
    gpu_barrier();
    let tb = unsafe { &*(h as *const TensorBuffer) };
    let axis_u = normalize_axis(axis as i32, tb.shape.len());
    let keepdim_b = keepdim != 0;
    let n = tb.shape[axis_u] as f32;
    let out_shape = reduce_axis_shape(&tb.shape, axis_u, keepdim_b);
    let out_len: usize = out_shape.iter().product::<usize>().max(1);
    let in_data = unsafe { std::slice::from_raw_parts(tb.buffer.contents().as_ptr() as *const f32, tb.len) };
    let ndims = tb.shape.len();
    let in_shape = tb.shape.clone();

    let mut mean_data = vec![0.0f32; out_len];
    reduce_elements(in_data, &in_shape, axis_u, keepdim_b, &out_shape, &mut mean_data, |a, b| a + b);
    for v in mean_data.iter_mut() { *v /= n; }

    let mut var_data = vec![0.0f32; out_len];
    for flat in 0..in_data.len() {
        let mut rem = flat;
        let mut in_idx = vec![0usize; ndims];
        for dim in (0..ndims).rev() {
            in_idx[dim] = rem % in_shape[dim];
            rem /= in_shape[dim];
        }
        let out_flat = reduce_out_flat(&in_idx, axis_u, keepdim_b, &out_shape);
        let diff = in_data[flat] - mean_data[out_flat];
        var_data[out_flat] += diff * diff;
    }
    for v in var_data.iter_mut() { *v /= n; }
    tensor_alloc_gpu(0, out_shape.as_ptr(), out_shape.len(), var_data.as_ptr())
}

// ── M17: reshape, permute, batched matmul ────────────────────────────────────

/// Normalize a list of raw dim args into a full permutation of `[0..rank)`.
/// 0 args  → reverse (rank must be 2, i.e. the no-arg transpose shorthand).
/// 2 args  → identity perm with those two axes swapped (any rank ≥ 2).
/// rank args → the full permutation itself; validated to be a bijection.
/// anything else → panic.
pub(crate) fn normalize_perm(raw: &[usize], rank: usize) -> Vec<usize> {
    match raw.len() {
        0 => {
            if rank != 2 {
                panic!("malus: transpose() with no dim args requires a 2-D tensor, got rank {rank}");
            }
            vec![1, 0]
        }
        2 => {
            let (i, j) = (raw[0], raw[1]);
            if i >= rank || j >= rank {
                panic!("malus: transpose axis {i} or {j} out of range for rank-{rank} tensor");
            }
            let mut perm: Vec<usize> = (0..rank).collect();
            perm.swap(i, j);
            perm
        }
        n if n == rank => {
            let mut seen = vec![false; rank];
            for &p in raw {
                if p >= rank {
                    panic!("malus: permute dim {p} out of range for rank-{rank} tensor");
                }
                if seen[p] {
                    panic!("malus: permute has duplicate dim {p}");
                }
                seen[p] = true;
            }
            raw.to_vec()
        }
        n => panic!(
            "malus: permute/transpose got {n} dim args for rank-{rank} tensor; \
             expected 0 (reverse 2-D), 2 (swap two axes), or {rank} (full permute)\n  \
             hint: transpose(t) reverses a 2-D tensor; \
             transpose(t,i,j) swaps two axes; \
             permute(t,p0..p_rank) reorders all axes"
        ),
    }
}

pub(crate) fn invert_perm(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}

/// Apply a fully-normalized permutation to a tensor.  No barrier — callers
/// that read GPU data must have already called gpu_barrier().
pub(crate) fn permute_by_perm(handle: i64, perm: &[usize]) -> i64 {
    let tb_in = unsafe { &*(handle as *const TensorBuffer) };
    let rank = tb_in.shape.len();
    assert_eq!(perm.len(), rank, "permute_by_perm: perm len {} != rank {}", perm.len(), rank);
    let out_shape: Vec<usize> = perm.iter().map(|&p| tb_in.shape[p]).collect();
    // Row-major strides for the input.
    let mut in_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        in_strides[i] = in_strides[i + 1] * tb_in.shape[i + 1];
    }
    let out_len: usize = out_shape.iter().product::<usize>().max(1);
    let in_data = unsafe { std::slice::from_raw_parts(tb_in.buffer.contents().as_ptr() as *const f32, tb_in.len) };
    let mut out_data = vec![0.0f32; out_len];
    // Row-major strides for the output.
    let mut out_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }
    for flat in 0..out_len {
        let mut rem = flat;
        let mut out_idx = vec![0usize; rank];
        for dim in (0..rank).rev() {
            out_idx[dim] = rem % out_shape[dim];
            rem /= out_shape[dim];
        }
        // out_dim d corresponds to in_dim perm[d], so in_idx[perm[d]] = out_idx[d].
        let mut in_flat = 0usize;
        for d in 0..rank {
            in_flat += out_idx[d] * in_strides[perm[d]];
        }
        out_data[flat] = in_data[in_flat];
    }
    tensor_alloc_gpu(0, out_shape.as_ptr(), out_shape.len(), out_data.as_ptr())
}

/// Zero-copy reshape: clone the MTLBuffer handle into a new TensorBuffer with
/// a different shape field.  No data copy — metal::Buffer::clone() is an
/// Obj-C retain on the same MTLBuffer.  Safe because M17 tensors are immutable.
pub(crate) fn reshape_to(handle: i64, new_shape: &[usize]) -> i64 {
    let tb = unsafe { &*(handle as *const TensorBuffer) };
    let new_tb = TensorBuffer {
        buffer: tb.buffer.clone(),
        dtype:  tb.dtype,
        len:    tb.len,
        shape:  new_shape.to_vec(),
        ref_count: std::sync::atomic::AtomicUsize::new(1),
    };
    Box::into_raw(Box::new(new_tb)) as i64
}

/// Public ABI: permute a tensor's axes.  Calls gpu_barrier() then
/// normalize_perm + permute_by_perm.
#[no_mangle]
pub extern "C" fn tensor_permute(handle: i64, perm_ptr: *const usize, ndims: usize) -> i64 {
    gpu_barrier();
    let tb_in = unsafe { &*(handle as *const TensorBuffer) };
    let raw: Vec<usize> = if ndims == 0 || perm_ptr.is_null() {
        vec![]
    } else {
        unsafe { std::slice::from_raw_parts(perm_ptr, ndims) }.to_vec()
    };
    let perm = normalize_perm(&raw, tb_in.shape.len());
    permute_by_perm(handle, &perm)
}

/// Public ABI: zero-copy reshape.  Panics on element-count mismatch (ADR-0013).
#[no_mangle]
pub extern "C" fn tensor_reshape(handle: i64, dims_ptr: *const usize, ndims: usize) -> i64 {
    let tb = unsafe { &*(handle as *const TensorBuffer) };
    let new_shape: Vec<usize> = unsafe { std::slice::from_raw_parts(dims_ptr, ndims) }.to_vec();
    let new_len: usize = new_shape.iter().product();
    if new_len != tb.len {
        panic!(
            "malus: reshape element count mismatch: \
             input shape {:?} has {} elements, \
             target shape {:?} would have {}\n  \
             input shape:  {:?}\n  target shape: {:?}",
            tb.shape, tb.len, new_shape, new_len, tb.shape, new_shape
        );
    }
    reshape_to(handle, &new_shape)
}

// ── M18: Transformer stdlib ───────────────────────────────────────────────────

/// Numerically stable softmax over one axis.  Called by tensor_softmax_axis
/// and tensor_cross_entropy (which shares this helper).
pub(crate) fn softmax_axis_cpu(in_data: &[f32], in_shape: &[usize], axis: usize) -> Vec<f32> {
    let axis_size = in_shape[axis];
    let outer: usize = in_shape[..axis].iter().product::<usize>().max(1);
    let inner: usize = in_shape[axis + 1..].iter().product::<usize>().max(1);
    let mut out = vec![0.0f32; in_data.len()];
    for o in 0..outer {
        for i in 0..inner {
            // Find max for numerical stability.
            let mut max_val = f32::NEG_INFINITY;
            for a in 0..axis_size {
                let v = in_data[o * axis_size * inner + a * inner + i];
                if v > max_val { max_val = v; }
            }
            // Compute exp(x - max) and sum.
            let mut sum = 0.0f32;
            for a in 0..axis_size {
                let e = (in_data[o * axis_size * inner + a * inner + i] - max_val).exp();
                out[o * axis_size * inner + a * inner + i] = e;
                sum += e;
            }
            // Normalize.
            for a in 0..axis_size {
                out[o * axis_size * inner + a * inner + i] /= sum;
            }
        }
    }
    out
}

#[no_mangle]
pub extern "C" fn tensor_softmax_axis(h: i64, axis: i64) -> i64 {
    gpu_barrier();
    let tb = unsafe { &*(h as *const TensorBuffer) };
    let axis_u = normalize_axis(axis as i32, tb.shape.len());
    let in_data = unsafe { std::slice::from_raw_parts(tb.buffer.contents().as_ptr() as *const f32, tb.len) };
    let out_data = softmax_axis_cpu(in_data, &tb.shape, axis_u);
    tensor_alloc_gpu(0, tb.shape.as_ptr(), tb.shape.len(), out_data.as_ptr())
}

/// Forward layernorm.  When `var_out` is non-null, writes the population
/// variance tensor (keepdim=1 at axis) to `*var_out` for use by the VJP
/// recorder.  Caller transfers ownership of the written handle to the tape.
#[no_mangle]
pub extern "C" fn tensor_layernorm_axis(h: i64, axis: i64, var_out: *mut i64) -> i64 {
    gpu_barrier();
    let tb = unsafe { &*(h as *const TensorBuffer) };
    let axis_u = normalize_axis(axis as i32, tb.shape.len());
    let axis_size = tb.shape[axis_u];
    let outer: usize = tb.shape[..axis_u].iter().product::<usize>().max(1);
    let inner: usize = tb.shape[axis_u + 1..].iter().product::<usize>().max(1);
    let in_data = unsafe { std::slice::from_raw_parts(tb.buffer.contents().as_ptr() as *const f32, tb.len) };

    let need_var = !var_out.is_null();
    let mut var_data = if need_var { vec![0.0f32; outer * inner] } else { vec![] };
    let mut norm_data = vec![0.0f32; tb.len];

    for o in 0..outer {
        for i in 0..inner {
            let mut mu = 0.0f32;
            for a in 0..axis_size {
                mu += in_data[o * axis_size * inner + a * inner + i];
            }
            mu /= axis_size as f32;
            let mut va = 0.0f32;
            for a in 0..axis_size {
                let diff = in_data[o * axis_size * inner + a * inner + i] - mu;
                va += diff * diff;
            }
            va /= axis_size as f32;
            let sigma = (va + 1e-5f32).sqrt();
            if need_var { var_data[o * inner + i] = va; }
            for a in 0..axis_size {
                let x_val = in_data[o * axis_size * inner + a * inner + i];
                norm_data[o * axis_size * inner + a * inner + i] = (x_val - mu) / sigma;
            }
        }
    }

    if need_var {
        // var shape has keepdim=1 at axis for easy broadcasting in the VJP.
        let mut var_shape = tb.shape.clone();
        var_shape[axis_u] = 1;
        let var_h = tensor_alloc_gpu(0, var_shape.as_ptr(), var_shape.len(), var_data.as_ptr());
        unsafe { *var_out = var_h; }
    }

    tensor_alloc_gpu(0, tb.shape.as_ptr(), tb.shape.len(), norm_data.as_ptr())
}

#[no_mangle]
pub extern "C" fn tensor_gelu(h: i64) -> i64 {
    gpu_barrier();
    let tb = unsafe { &*(h as *const TensorBuffer) };
    let in_data = unsafe { std::slice::from_raw_parts(tb.buffer.contents().as_ptr() as *const f32, tb.len) };
    const C0: f32 = 0.7978845608;
    const C1: f32 = 0.044715;
    let out_data: Vec<f32> = in_data.iter().map(|&x| {
        let inner = C0 * (x + C1 * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }).collect();
    tensor_alloc_gpu(0, tb.shape.as_ptr(), tb.shape.len(), out_data.as_ptr())
}

/// Forward cross-entropy.  Computes softmax(logits) over the class axis (1),
/// then -log(softmax[i, target[i]]) averaged over N.
/// When `softmax_out` is non-null, writes the softmax tensor handle to
/// `*softmax_out` for use by the VJP recorder.  Caller transfers ownership.
#[no_mangle]
pub extern "C" fn tensor_cross_entropy(logits: i64, targets: i64, softmax_out: *mut i64) -> i64 {
    gpu_barrier();
    let logits_tb = unsafe { &*(logits as *const TensorBuffer) };
    let targets_tb = unsafe { &*(targets as *const TensorBuffer) };
    if logits_tb.shape.len() != 2 {
        panic!("malus: cross_entropy logits must be rank-2 [N, C], got shape {:?}", logits_tb.shape);
    }
    if targets_tb.shape.len() != 1 {
        panic!("malus: cross_entropy targets must be rank-1 [N], got shape {:?}", targets_tb.shape);
    }
    let n = logits_tb.shape[0];
    let c = logits_tb.shape[1];
    if targets_tb.shape[0] != n {
        panic!("malus: cross_entropy logits N={} but targets has {} elements", n, targets_tb.shape[0]);
    }
    let logits_data = unsafe {
        std::slice::from_raw_parts(logits_tb.buffer.contents().as_ptr() as *const f32, logits_tb.len)
    };

    let sm_data = softmax_axis_cpu(logits_data, &logits_tb.shape, 1);

    let idx_buf = targets_tb.buffer.contents().as_ptr() as *const u8;
    let mut loss = 0.0f32;
    for i in 0..n {
        let t = read_int_index(idx_buf, i, targets_tb.dtype);
        if t >= c {
            panic!("malus: cross_entropy target index {} out of range for C={}", t, c);
        }
        loss -= sm_data[i * c + t].ln();
    }
    loss /= n as f32;

    if !softmax_out.is_null() {
        let sm_shape = logits_tb.shape.clone();
        let sm_h = tensor_alloc_gpu(0, sm_shape.as_ptr(), sm_shape.len(), sm_data.as_ptr());
        unsafe { *softmax_out = sm_h; }
    }

    let loss_shape = [1usize];
    tensor_alloc_gpu(0, loss_shape.as_ptr(), 1, [loss].as_ptr())
}

/// Returns a [T, T] causal mask: 0.0 on and below the diagonal, -1e9 above.
#[no_mangle]
pub extern "C" fn tensor_causal_mask(t_size: i64) -> i64 {
    let t = t_size as usize;
    let mut data = vec![0.0f32; t * t];
    for i in 0..t {
        for j in 0..t {
            if j > i { data[i * t + j] = -1e9; }
        }
    }
    let shape = [t, t];
    tensor_alloc_gpu(0, shape.as_ptr(), 2, data.as_ptr())
}

// ── M19: index-tensor helpers ─────────────────────────────────────────────────

/// Read element `i` of an integer (i32 or i64) buffer as `usize`.
/// Panics if dtype is not an integer index type.
fn read_int_index(buf: *const u8, i: usize, dtype: Dtype) -> usize {
    match dtype {
        Dtype::I32 => unsafe { *(buf.add(i * 4) as *const i32) as usize }
        Dtype::I64 => unsafe { *(buf.add(i * 8) as *const i64) as usize }
        _ => panic!("malus: integer index tensor must be Tensor<i32> or Tensor<i64>, got {:?}", dtype),
    }
}

/// Embedding lookup: out[t] = weight[indices[t]], shape [T, D].
/// weight must be [V, D], indices must be [T] of dtype i32 or i64.
#[no_mangle]
pub extern "C" fn tensor_embedding(weight: i64, indices: i64) -> i64 {
    gpu_barrier();
    let w = unsafe { &*(weight as *const TensorBuffer) };
    let idx = unsafe { &*(indices as *const TensorBuffer) };
    if w.shape.len() != 2 {
        panic!("malus: embedding weight must be rank-2 [V, D], got shape {:?}", w.shape);
    }
    if idx.shape.len() != 1 {
        panic!("malus: embedding indices must be rank-1 [T], got shape {:?}", idx.shape);
    }
    let vocab_size  = w.shape[0];
    let embed_dim   = w.shape[1];
    let seq_len     = idx.shape[0];
    let weight_data = unsafe {
        std::slice::from_raw_parts(w.buffer.contents().as_ptr() as *const f32, w.len)
    };
    let idx_buf = idx.buffer.contents().as_ptr() as *const u8;
    let mut out = vec![0.0f32; seq_len * embed_dim];
    for t in 0..seq_len {
        let row = read_int_index(idx_buf, t, idx.dtype);
        if row >= vocab_size {
            panic!("malus: embedding index {} out of range [0, {})", row, vocab_size);
        }
        out[t * embed_dim..(t + 1) * embed_dim]
            .copy_from_slice(&weight_data[row * embed_dim..(row + 1) * embed_dim]);
    }
    let out_shape = [seq_len, embed_dim];
    tensor_alloc_gpu(0, out_shape.as_ptr(), 2, out.as_ptr())
}

/// Scatter-add: dweight[indices[t]] += dout[t, :] for t in 0..T.
/// dout is [T, D], indices is [T] (i32/i64), vocab_size is V.
/// Returns a [V, D] tensor.
pub(crate) fn tensor_scatter_add(dout: i64, indices: i64, vocab_size: i64) -> i64 {
    let dout_tb = unsafe { &*(dout as *const TensorBuffer) };
    let idx_tb  = unsafe { &*(indices as *const TensorBuffer) };
    let seq_len   = dout_tb.shape[0];
    let embed_dim = dout_tb.shape[1];
    let v         = vocab_size as usize;
    let dout_data = unsafe {
        std::slice::from_raw_parts(dout_tb.buffer.contents().as_ptr() as *const f32, dout_tb.len)
    };
    let idx_buf = idx_tb.buffer.contents().as_ptr() as *const u8;
    let mut dw = vec![0.0f32; v * embed_dim];
    for t in 0..seq_len {
        let row = read_int_index(idx_buf, t, idx_tb.dtype);
        for d in 0..embed_dim {
            dw[row * embed_dim + d] += dout_data[t * embed_dim + d];
        }
    }
    let shape = [v, embed_dim];
    tensor_alloc_gpu(0, shape.as_ptr(), 2, dw.as_ptr())
}

// ── M19: Philox4x32-10 counter-based RNG ─────────────────────────────────────

const PHILOX_M4X32_0: u32 = 0xD2511F53;
const PHILOX_M4X32_1: u32 = 0xCD9E8D57;
const PHILOX_W32_0:   u32 = 0x9E3779B9;
const PHILOX_W32_1:   u32 = 0xBB67AE85;

fn mulhilo32(a: u32, b: u32) -> (u32, u32) {
    let p = (a as u64).wrapping_mul(b as u64);
    ((p >> 32) as u32, p as u32)
}

fn philox4x32_10(mut c: [u32; 4], mut k: [u32; 2]) -> [u32; 4] {
    for _ in 0..10 {
        let (h0, l0) = mulhilo32(c[0], PHILOX_M4X32_0);
        let (h1, l1) = mulhilo32(c[2], PHILOX_M4X32_1);
        c = [h1 ^ c[1] ^ k[0], l1, h0 ^ c[3] ^ k[1], l0];
        k[0] = k[0].wrapping_add(PHILOX_W32_0);
        k[1] = k[1].wrapping_add(PHILOX_W32_1);
    }
    c
}

thread_local! {
    static RANDN_CALL_COUNTER: Cell<u64> = const { Cell::new(0) };
}

/// randn(d0, d1, ...) — standard-normal tensor, CPU-generated, GPU-placed.
/// Algorithm: Philox4x32-10 keyed by a per-call thread-local counter, then
/// Box-Muller transform. Fixed default seed (counter starts at 0 per thread).
#[no_mangle]
pub extern "C" fn tensor_randn(shape_ptr: *const usize, ndims: usize) -> i64 {
    let shape = unsafe { std::slice::from_raw_parts(shape_ptr, ndims).to_vec() };
    let n: usize = shape.iter().product();
    let call_idx = RANDN_CALL_COUNTER.with(|c| { let v = c.get(); c.set(v + 1); v });
    let key = [call_idx as u32, (call_idx >> 32) as u32];
    let mut data = vec![0.0f32; n];
    let n_pairs = (n + 1) / 2;
    for k in 0..n_pairs {
        let ctr = [k as u32, (k >> 32) as u32, 0u32, 0u32];
        let r = philox4x32_10(ctr, key);
        let u1 = (r[0] as f64 + 0.5) / 4_294_967_296.0;
        let u2 = (r[1] as f64 + 0.5) / 4_294_967_296.0;
        let mag = (-2.0 * u1.ln()).sqrt();
        let z0  = (mag * (2.0 * std::f64::consts::PI * u2).cos()) as f32;
        let z1  = (mag * (2.0 * std::f64::consts::PI * u2).sin()) as f32;
        data[2 * k] = z0;
        if 2 * k + 1 < n { data[2 * k + 1] = z1; }
    }
    tensor_alloc_gpu(0, shape.as_ptr(), ndims, data.as_ptr())
}

// ── VJP helpers (pub(crate) for tape.rs) ─────────────────────────────────────

/// Expand `h` to `out_shape` using NumPy broadcast semantics.
/// Returns a retained handle (caller must release). Identity (retain only) when shapes match.
pub(crate) fn broadcast_to_shape(h: i64, out_shape: &[usize]) -> i64 {
    let t = unsafe { &*(h as *const TensorBuffer) };
    if t.shape.as_slice() == out_shape {
        tensor_retain(h);
        return h;
    }
    let n = out_shape.len();
    let mut padded = vec![1usize; n - t.shape.len()];
    padded.extend_from_slice(&t.shape);
    let in_data = unsafe { std::slice::from_raw_parts(t.buffer.contents().as_ptr() as *const f32, t.len) };
    let out_len: usize = out_shape.iter().product();
    let mut out_data = vec![0.0f32; out_len.max(1)];
    for flat in 0..out_len {
        let mut rem = flat;
        let mut out_idx = vec![0usize; n];
        for dim in (0..n).rev() {
            out_idx[dim] = rem % out_shape[dim];
            rem /= out_shape[dim];
        }
        let mut in_flat = 0usize;
        for dim in 0..n {
            in_flat = in_flat * padded[dim] + (out_idx[dim] % padded[dim]);
        }
        out_data[flat] = in_data[in_flat];
    }
    tensor_alloc_gpu(0, out_shape.as_ptr(), out_shape.len(), out_data.as_ptr())
}

/// Sum `grad` down to `target_shape` (the operand shape before broadcasting).
/// Returns a retained handle. Identity (retain only) when shapes already match.
pub(crate) fn sum_to_shape(grad: i64, target_shape: &[usize]) -> i64 {
    let t = unsafe { &*(grad as *const TensorBuffer) };
    if t.shape.as_slice() == target_shape {
        tensor_retain(grad);
        return grad;
    }
    let n = t.shape.len();
    let n_target = target_shape.len();
    let mut padded = vec![1usize; n - n_target];
    padded.extend_from_slice(target_shape);
    let in_data = unsafe { std::slice::from_raw_parts(t.buffer.contents().as_ptr() as *const f32, t.len) };
    let out_len: usize = target_shape.iter().product::<usize>().max(1);
    let mut out_data = vec![0.0f32; out_len];
    for flat in 0..t.len {
        let mut rem = flat;
        let mut in_idx = vec![0usize; n];
        for dim in (0..n).rev() {
            in_idx[dim] = rem % t.shape[dim];
            rem /= t.shape[dim];
        }
        let mut out_flat = 0usize;
        for dim in 0..n {
            out_flat = out_flat * padded[dim] + (in_idx[dim] % padded[dim]);
        }
        out_data[out_flat] += in_data[flat];
    }
    tensor_alloc_gpu(0, target_shape.as_ptr(), target_shape.len(), out_data.as_ptr())
}

/// Insert a size-1 dimension at `axis` (reshape; no data copy cost beyond alloc).
/// Returns an owned handle (refcount=1).
pub(crate) fn unsqueeze_at(h: i64, axis: usize) -> i64 {
    let t = unsafe { &*(h as *const TensorBuffer) };
    let data = unsafe { std::slice::from_raw_parts(t.buffer.contents().as_ptr() as *const f32, t.len) };
    let mut new_shape = t.shape.clone();
    new_shape.insert(axis, 1);
    tensor_alloc_gpu(0, new_shape.as_ptr(), new_shape.len(), data.as_ptr())
}

// ── Kernel dispatch ───────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn kernel_dispatch(kernel_id: u64, handles: *const i64, count: usize) -> i64 {
    if count < 1 || handles.is_null() {
        panic!("malus: kernel_dispatch requires at least one input handle");
    }

    let ctx = context();

    let pipeline = {
        let pipelines = ctx.pipelines.lock().unwrap();
        pipelines.get(&kernel_id)
            .expect(&format!("malus: kernel_id {} not registered", kernel_id))
            .clone()
    };

    let inputs: Vec<&TensorBuffer> = (0..count)
        .map(|i| unsafe { &*(handles.add(i).read() as *const TensorBuffer) })
        .collect();

    let first = &inputs[0];
    let out_dtype = first.dtype;
    let out_shape = first.shape.clone();

    let output_handle = tensor_alloc_gpu(
        out_dtype.to_tag(),
        out_shape.as_ptr(),
        out_shape.len(),
        std::ptr::null(),
    );
    let output_tb = unsafe { &*(output_handle as *const TensorBuffer) };

    // Defensive guard: a zero-length output means there is nothing to dispatch.
    // Encoding a dispatchThreads with grid_size = (0,1,1) aborts the Metal encoder.
    if output_tb.len == 0 {
        return output_handle;
    }

    let mut guard = ctx.current_command_buffer.lock().unwrap();
    if guard.is_none() {
        let cmd_buf = ctx.command_queue
            .commandBuffer()
            .expect("malus: failed to create MTLCommandBuffer");
        *guard = Some(cmd_buf);
    }
    let cmd_buf = guard.as_ref().unwrap();

    let encoder = cmd_buf
        .computeCommandEncoder()
        .expect("malus: failed to create MTLComputeCommandEncoder");
    encoder.setComputePipelineState(&*pipeline);

    for (i, input) in inputs.iter().enumerate() {
        unsafe { encoder.setBuffer_offset_atIndex(Some(&*input.buffer), 0, i) };
    }
    unsafe { encoder.setBuffer_offset_atIndex(Some(&*output_tb.buffer), 0, count) };

    let out_len = output_tb.len;
    let grid_size = MTLSize { width: out_len, height: 1, depth: 1 };
    let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
    let threadgroup_size = MTLSize {
        width: max_threads.min(out_len),
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    encoder.endEncoding();

    output_handle
}
