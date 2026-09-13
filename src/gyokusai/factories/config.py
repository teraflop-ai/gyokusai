GENERATION_KWARGS = {
    "trust_remote_code": True,
    "kv_cache_dtype": "fp8_e4m3",
    "attention_backend": "flashinfer",
    "mamba_full_memory_ratio": 1.87,
    "mamba_radix_cache_strategy": "extra_buffer_lazy",
    "mamba_ssm_dtype": "bfloat16",
    "allow_auto_truncate": True,
    "grammar_backend": "xgrammar",
    "constrained_json_disable_any_whitespace": True,
}
