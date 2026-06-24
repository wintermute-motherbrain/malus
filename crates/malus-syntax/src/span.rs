#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FileId(pub u32);

/// Half-open byte range [start, end) within a source file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub file: FileId,
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub fn new(file: FileId, start: usize, end: usize) -> Self {
        Self { file, start: start as u32, end: end as u32 }
    }

    pub fn at(file: FileId, pos: usize) -> Self {
        Self::new(file, pos, pos)
    }
}
