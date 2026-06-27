use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use foreign_types::{ForeignType, ForeignTypeRef};
use metal::{
    CommandBuffer, CommandQueue, CompileOptions, ComputePipelineState,
    Device, MTLResourceOptions, MTLSize, NSUInteger,
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
    device: Device,
    command_queue: CommandQueue,
    current_command_buffer: Mutex<Option<CommandBuffer>>,
    pipelines: Mutex<HashMap<u64, ComputePipelineState>>,
}

static CONTEXT: OnceLock<MetalContext> = OnceLock::new();

fn context() -> &'static MetalContext {
    CONTEXT.get_or_init(|| {
        let device = Device::system_default()
            .expect("malus: no Metal device available");
        let command_queue = device.new_command_queue();
        MetalContext {
            device,
            command_queue,
            current_command_buffer: Mutex::new(None),
            pipelines: Mutex::new(HashMap::new()),
        }
    })
}

pub struct TensorBuffer {
    pub buffer: metal::Buffer,
    pub dtype: Dtype,
    pub len: usize,
    pub shape: Vec<usize>,
}

// ── Runtime init: compile all MSL kernels ─────────────────────────────────────

pub fn runtime_init(registry: &HashMap<u64, String>) {
    let ctx = context();
    let mut pipelines = ctx.pipelines.lock().unwrap();

    for (id, source) in registry {
        let options = CompileOptions::new();
        let library = ctx.device
            .new_library_with_source(source, &options)
            .unwrap_or_else(|e| panic!("malus: MSL compilation failed for kernel {}: {}", id, e));
        let func_name = format!("malus_kernel_{}", id);
        let function = library
            .get_function(&func_name, None)
            .unwrap_or_else(|e| panic!("malus: kernel function '{}' not found: {}", func_name, e));
        let pipeline = ctx.device
            .new_compute_pipeline_state_with_function(&function)
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
    if dt != Dtype::F32 {
        panic!("malus: non-f32 dtypes not yet implemented (got dtype tag {dtype})");
    }
    let shape = unsafe { std::slice::from_raw_parts(shape_ptr, ndims).to_vec() };
    let n: usize = shape.iter().product();
    let byte_len = n * dt.element_size();

    let ctx = context();
    let buffer = ctx.device.new_buffer(
        byte_len as u64,
        MTLResourceOptions::StorageModeShared,
    );

    if !data.is_null() && n > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(
                data as *const u8,
                buffer.contents() as *mut u8,
                byte_len,
            );
        }
    }

    let tb = Box::new(TensorBuffer { buffer, dtype: dt, len: n, shape });
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

#[no_mangle]
pub extern "C" fn tensor_free(handle: i64) {
    if handle == 0 {
        return;
    }
    unsafe {
        drop(Box::from_raw(handle as *mut TensorBuffer));
    }
}

#[no_mangle]
pub extern "C" fn tensor_print(handle: i64) {
    if handle == 0 {
        print!("[]");
        return;
    }
    let tb = unsafe { &*(handle as *const TensorBuffer) };
    let ptr = tb.buffer.contents() as *const f32;
    let slice = unsafe { std::slice::from_raw_parts(ptr, tb.len) };
    print!("[");
    for (i, v) in slice.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{v}");
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
        cmd_buf.wait_until_completed();
    }
}

// ── Eager CPU ops (matmul, transpose, sum) ────────────────────────────────────
//
// These commit any pending GPU work before reading shared buffers on the CPU,
// then compute in plain Rust and return a ready (non-pending) output tensor.
// Migration to MPS is deferred to post-V1 (see ADR-0012).

#[no_mangle]
pub extern "C" fn tensor_matmul(handle_a: i64, handle_b: i64) -> i64 {
    gpu_barrier();
    let a = unsafe { &*(handle_a as *const TensorBuffer) };
    let b = unsafe { &*(handle_b as *const TensorBuffer) };

    if a.shape.len() != 2 {
        panic!("malus: tensor_matmul requires a 2-D tensor, got shape {:?} for first arg", a.shape);
    }
    if b.shape.len() != 2 {
        panic!("malus: tensor_matmul requires a 2-D tensor, got shape {:?} for second arg", b.shape);
    }
    let (m, k) = (a.shape[0], a.shape[1]);
    let (k2, n) = (b.shape[0], b.shape[1]);
    if k != k2 {
        panic!("malus: tensor_matmul dim mismatch: {:?} @ {:?}", a.shape, b.shape);
    }

    let a_data = unsafe { std::slice::from_raw_parts(a.buffer.contents() as *const f32, a.len) };
    let b_data = unsafe { std::slice::from_raw_parts(b.buffer.contents() as *const f32, b.len) };

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

#[no_mangle]
pub extern "C" fn tensor_transpose(handle: i64) -> i64 {
    gpu_barrier();
    let tb = unsafe { &*(handle as *const TensorBuffer) };

    if tb.shape.len() != 2 {
        panic!("malus: tensor_transpose requires a 2-D tensor, got shape {:?}", tb.shape);
    }
    let m = tb.shape[0];
    let n = tb.shape[1];

    let in_data = unsafe { std::slice::from_raw_parts(tb.buffer.contents() as *const f32, tb.len) };
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
    let data = unsafe { std::slice::from_raw_parts(tb.buffer.contents() as *const f32, tb.len) };
    let total: f32 = data.iter().sum();
    let shape = [1usize];
    tensor_alloc_gpu(0, shape.as_ptr(), 1, &total)
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

    let mut guard = ctx.current_command_buffer.lock().unwrap();
    if guard.is_none() {
        let cmd_buf_ref = ctx.command_queue.new_command_buffer();
        let retained: *mut metal::MTLCommandBuffer = unsafe {
            msg_send![cmd_buf_ref.as_ptr(), retain]
        };
        let cmd_buf = unsafe { CommandBuffer::from_ptr(retained) };
        *guard = Some(cmd_buf);
    }
    let cmd_buf = guard.as_ref().unwrap();

    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    for (i, input) in inputs.iter().enumerate() {
        encoder.set_buffer(i as NSUInteger, Some(&input.buffer), 0);
    }
    encoder.set_buffer(count as NSUInteger, Some(&output_tb.buffer), 0);

    let out_len = output_tb.len;
    let grid_size = MTLSize::new(out_len as NSUInteger, 1, 1);
    let max_threads = pipeline.max_total_threads_per_threadgroup();
    let threadgroup_size = MTLSize::new(
        max_threads.min(out_len as NSUInteger),
        1,
        1,
    );
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();

    output_handle
}
