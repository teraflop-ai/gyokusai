GENERATION_KWARGS = {
    "trust_remote_code": True,
    "kv_cache_dtype": "fp8_e4m3",
    "attention_backend": "flashinfer",
    "mamba_ssm_dtype": "bfloat16",
    "grammar_backend": "xgrammar",
    "constrained_json_disable_any_whitespace": True,
    "allow_auto_truncate": True,
    "log_level": "info",
}