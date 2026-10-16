import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import make_tokenizers

EDGE_CASES = [
    'п’ять, п\'ять, зв’язок, з\'їзд, бур’ян',
    'Ґанок, їжак, єнот, ідея, щирість, м’якість.',
    'Київ — Україна! «Так», “так”, „так“.',
    'Ціна: 12 345,67 ₴; знижка −20%; 3,14.',
    'Emoji: 💙💛 🔥 🦾; латинка: HTML, AI, ChatGPT.',
    'Довге слово: найвисококваліфікованішими.',
    'Поєднання літер: й і̆ ё. Не нормалізувати.',
    '  Початкові й кінцеві пробіли  ',
]


def test_existing_models_lossless():
    demo = Path(__file__).resolve().parents[1]/'runs/demo'
    if not (demo/'uk_only.model').exists():
        return
    models = make_tokenizers([f'uk:{demo}/uk_only.model',f'mix:{demo}/balanced_uk_en.model'], [])
    for name, (encode,decode,_,_) in models.items():
        for item in EDGE_CASES:
            assert decode(encode(item)) == item, (name,repr(item),repr(decode(encode(item))))
