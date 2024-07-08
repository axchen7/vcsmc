def custom_module_repr(extra_args: dict):
    return "\n".join([f"({k}): {v}" for k, v in extra_args.items()])
