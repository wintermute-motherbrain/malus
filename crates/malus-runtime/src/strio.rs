// M22 — string I/O runtime functions.
//
// Strings at the malus ABI boundary are opaque i64 handles pointing at a
// heap-allocated `StrBox { ptr: *const u8, len: usize }`.  All StrBoxes are
// leaked (whole-program lifetime per ADR-0018); none are freed.

#[repr(C)]
pub struct StrBox {
    pub ptr: *const u8,
    pub len: usize,
}

unsafe impl Send for StrBox {}
unsafe impl Sync for StrBox {}

#[inline]
unsafe fn deref(handle: i64) -> &'static StrBox {
    &*(handle as *const StrBox)
}

/// Create a heap-allocated, leaked StrBox from a raw pointer and length.
/// Used by codegen to materialise `Lit::Str` values at JIT time.
#[no_mangle]
pub extern "C" fn malus_str_box(ptr: *const u8, len: usize) -> i64 {
    let b = Box::new(StrBox { ptr, len });
    Box::into_raw(b) as i64
}

/// Read a UTF-8 text file from disk and return its contents as a new StrBox
/// handle.  Panics (ADR-0006 panic-only error model) if the file cannot be
/// read.  Both the content buffer and the StrBox are leaked.
#[no_mangle]
pub extern "C" fn malus_read_file(path_handle: i64) -> i64 {
    let sb = unsafe { deref(path_handle) };
    let path_bytes = unsafe { std::slice::from_raw_parts(sb.ptr, sb.len) };
    let path = std::str::from_utf8(path_bytes)
        .unwrap_or_else(|_| panic!("malus: read_file: path is not valid UTF-8"));
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("malus: read_file({:?}): {}", path, e));
    let bytes = content.into_bytes();
    let len = bytes.len();
    let ptr = Box::into_raw(bytes.into_boxed_slice()) as *const u8;
    let sb2 = Box::new(StrBox { ptr, len });
    Box::into_raw(sb2) as i64
}

/// Return the byte length of the string (not the number of Unicode scalars).
/// Matches Python len() / Rust .len() on str.  For ASCII text this equals
/// the character count.  Returns i64 to match malus's native integer width.
#[no_mangle]
pub extern "C" fn malus_str_len(handle: i64) -> i64 {
    let sb = unsafe { deref(handle) };
    sb.len as i64
}

/// Return the i-th Unicode scalar value (codepoint) in the string, or -1 if
/// idx is out of range.  O(n) in the character position — suitable for small
/// vocabularies (the tiny-Shakespeare char set is ASCII-only).
/// Both index and return value are i64 to match malus's native integer width.
#[no_mangle]
pub extern "C" fn malus_str_char_at(handle: i64, idx: i64) -> i64 {
    let sb = unsafe { deref(handle) };
    let bytes = unsafe { std::slice::from_raw_parts(sb.ptr, sb.len) };
    let s = std::str::from_utf8(bytes).unwrap_or("");
    match s.chars().nth(idx as usize) {
        Some(c) => c as i64,
        None => -1,
    }
}

/// Encode a Unicode codepoint as a UTF-8 StrBox.  Leaks the result.
/// Invalid codepoints are replaced with U+FFFD.
/// Takes and returns i64 to match malus's native integer width.
#[no_mangle]
pub extern "C" fn malus_str_from_char(c: i64) -> i64 {
    let ch = char::from_u32(c as u32).unwrap_or('\u{FFFD}');
    let mut tmp = [0u8; 4];
    let encoded = ch.encode_utf8(&mut tmp);
    let bytes = encoded.as_bytes().to_vec();
    let len = bytes.len();
    let ptr = Box::into_raw(bytes.into_boxed_slice()) as *const u8;
    let sb = Box::new(StrBox { ptr, len });
    Box::into_raw(sb) as i64
}
