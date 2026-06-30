use malus_syntax::{ast::Item, span::FileId};

static GELU_SRC:           &str = include_str!("../stdlib/gelu.ml");
static SOFTMAX_SRC:        &str = include_str!("../stdlib/softmax.ml");
static LAYERNORM_SRC:      &str = include_str!("../stdlib/layernorm.ml");
static EMBEDDING_SRC:      &str = include_str!("../stdlib/embedding.ml");
static PERMUTE_SRC:        &str = include_str!("../stdlib/permute.ml");
static BROADCAST_SRC:      &str = include_str!("../stdlib/broadcast_binop.ml");
static REDUCE_SUM_SRC:     &str = include_str!("../stdlib/reduce_sum.ml");
static REDUCE_MEAN_SRC:    &str = include_str!("../stdlib/reduce_mean.ml");
static REDUCE_MAX_SRC:     &str = include_str!("../stdlib/reduce_max.ml");
static REDUCE_VAR_SRC:     &str = include_str!("../stdlib/reduce_var.ml");
// cross_entropy.ml calls __softmax_fwd and __reduce_mean_fwd; must be parsed
// after the above but sema's two-pass sig collection makes item order irrelevant.
static CROSS_ENTROPY_SRC:  &str = include_str!("../stdlib/cross_entropy.ml");

const FILES: &[(&str, &str)] = &[
    ("stdlib/gelu.ml",          GELU_SRC),
    ("stdlib/softmax.ml",       SOFTMAX_SRC),
    ("stdlib/layernorm.ml",     LAYERNORM_SRC),
    ("stdlib/embedding.ml",     EMBEDDING_SRC),
    ("stdlib/permute.ml",       PERMUTE_SRC),
    ("stdlib/broadcast_binop.ml", BROADCAST_SRC),
    ("stdlib/reduce_sum.ml",    REDUCE_SUM_SRC),
    ("stdlib/reduce_mean.ml",   REDUCE_MEAN_SRC),
    ("stdlib/reduce_max.ml",    REDUCE_MAX_SRC),
    ("stdlib/reduce_var.ml",    REDUCE_VAR_SRC),
    ("stdlib/cross_entropy.ml", CROSS_ENTROPY_SRC),
];

pub fn stdlib_items() -> Vec<Item> {
    let mut items: Vec<Item> = Vec::new();
    for (i, (_path, src)) in FILES.iter().enumerate() {
        let file_id = FileId(1000 + i as u32);
        let program = malus_syntax::parse(file_id, src)
            .unwrap_or_else(|e| panic!("malus-stdlib: parse error in {}: {}", _path, e));
        items.extend(program.items);
    }
    items
}
