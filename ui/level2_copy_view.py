def format_variants(variants):
    output = ""
    for name, content in variants.items():
        output += f"\n\n=== {name.upper()} VARIANT ===\n"
        output += content
    return output
