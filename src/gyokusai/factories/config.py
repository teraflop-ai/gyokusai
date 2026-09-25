GENERATION_KWARGS = {
    "trust_remote_code": True,
    "kv_cache_dtype": "fp8_e4m3",
    "attention_backend": "flashinfer",
    "mamba_ssm_cache_dtype": "bfloat16",
    "enable_prefix_caching": True,
    "structured_outputs_config": {
        "backend": "xgrammar",
        "disable_any_whitespace": True,
    },
}
