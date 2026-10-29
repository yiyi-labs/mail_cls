"""Show the numeric boundary honestly instead of rounding it to certainty."""
def score_text(value):
    if value == 1.0:return '≈ 1（浮点饱和）'
    if value == 0.0:return '≈ 0（浮点饱和）'
    if value > .999999:return f'1 − {1-value:.3e}'
    if value < .000001:return f'{value:.6e}'
    return f'{value:.9f}'
