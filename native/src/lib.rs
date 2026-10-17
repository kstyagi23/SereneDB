use std::alloc::{alloc, dealloc, Layout};
use std::arch::wasm32::*;

#[unsafe(no_mangle)]
pub extern "C" fn allocate(bytes: usize) -> *mut u8 {
    let layout = Layout::from_size_align(bytes.max(1), 16).unwrap();
    unsafe { alloc(layout) }
}

#[unsafe(no_mangle)]
pub extern "C" fn release(pointer: *mut u8, bytes: usize) {
    if !pointer.is_null() {
        let layout = Layout::from_size_align(bytes.max(1), 16).unwrap();
        unsafe { dealloc(pointer, layout) };
    }
}

// Rows and the query are float32; output is negative dot or squared L2.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn scan(
    rows: *const f32,
    query: *const f32,
    output: *mut f32,
    count: usize,
    dimension: usize,
    euclidean: u32,
) {
    for row in 0..count {
        let offset = row * dimension;
        let mut accumulator = f32x4_splat(0.0);
        let mut i = 0;
        while i + 4 <= dimension {
            let a = unsafe { v128_load(rows.add(offset + i) as *const v128) };
            let b = unsafe { v128_load(query.add(i) as *const v128) };
            let product = if euclidean == 1 {
                let difference = f32x4_sub(a, b);
                f32x4_mul(difference, difference)
            } else {
                f32x4_mul(a, b)
            };
            accumulator = f32x4_add(accumulator, product);
            i += 4;
        }
        let mut sum = f32x4_extract_lane::<0>(accumulator)
            + f32x4_extract_lane::<1>(accumulator)
            + f32x4_extract_lane::<2>(accumulator)
            + f32x4_extract_lane::<3>(accumulator);
        while i < dimension {
            let a = unsafe { *rows.add(offset + i) };
            let b = unsafe { *query.add(i) };
            sum += if euclidean == 1 { (a - b) * (a - b) } else { a * b };
            i += 1;
        }
        unsafe { *output.add(row) = if euclidean == 1 { sum } else { -sum } };
    }
}
