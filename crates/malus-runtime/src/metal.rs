use std::sync::OnceLock;

use metal::{
    Device, CommandQueue, Buffer, MTLResourceOptions,
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
}

static CONTEXT: OnceLock<MetalContext> = OnceLock::new();

fn context() -> &'static MetalContext {
    CONTEXT.get_or_init(|| {
        let device = Device::system_default()
            .expect("malus: no Metal device available");
        let command_queue = device.new_command_queue();
        MetalContext { device, command_queue }
    })
}

pub struct TensorBuffer {
    pub buffer: Buffer,
    pub dtype: Dtype,
    pub len: usize,
}

#[no_mangle]
pub extern "C" fn tensor_alloc_gpu(dtype: i32, len: i64, data: *const f32) -> i64 {
    let dt = Dtype::from_tag(dtype);
    if dt != Dtype::F32 {
        panic!("malus: non-f32 dtypes not yet implemented (got dtype tag {dtype})");
    }
    let n = len as usize;
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

    let tb = Box::new(TensorBuffer { buffer, dtype: dt, len: n });
    Box::into_raw(tb) as i64
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
pub extern "C" fn gpu_barrier() {
    let ctx = context();
    let cmd = ctx.command_queue.new_command_buffer();
    cmd.commit();
    cmd.wait_until_completed();
}

#[no_mangle]
pub extern "C" fn kernel_dispatch(_name: *const u8, handles: *const i64, n: i32) -> i64 {
    if n < 1 || handles.is_null() {
        panic!("malus: kernel_dispatch requires at least one input handle");
    }
    let first = unsafe { &*(handles.read() as *const TensorBuffer) };
    tensor_alloc_gpu(0, first.len as i64, std::ptr::null())
}
