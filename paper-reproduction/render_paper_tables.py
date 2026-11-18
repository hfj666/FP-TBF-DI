"""Render numbered manuscript tables and retain the editable table data."""
import re
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PAPER = Path(__file__).resolve().parent / 'paper'
MANUSCRIPT = PAPER / 'FP_TBF_DI_中文论文_完整投稿稿.md'
ASSETS = PAPER / 'figures' / 'submission_tables'
TABLE_SOURCE = PAPER / 'figures' / 'submission_table_source.json'
PATTERN = re.compile(
    r'(?P<caption>^\*\*表(?P<number>\d+) [^\n]+\*\*)\n\n'
    r'(?:!\[[^\n]*\]\([^\n]*\)\n\n<details>\n<summary>查看表格数据</summary>\n\n)?'
    r'(?P<table>(?:^\|[^\n]*\|\n)+)(?:\n</details>\n)?', re.M)
LABELS = dict(line.split('|', 1) for line in
              (PAPER / 'figures/submission_table_labels_en.txt').read_text(encoding='utf-8').splitlines()
              if line.strip())
CAPTIONS = (
    'Dataset and Evaluation Units',
    'Sequential Performance of FP-TBF-DI by Configuration',
    'Confusion Matrix for Sequential Predictions by FP-TBF-DI',
    'Comparison of Individual Branches and Reference Models',
    'Comparison of Two-Branch and Three-Branch Fusion',
    'Paired Changes in Prediction Correctness for FP-TBF-DI',
    'Fixed Holdout Performance at Different Training Sizes',
    'Sequential Performance for Previously Seen and Unseen Score Vectors',
    'Matched Holdout Results with 849 Training Records',
    'Rule Selection at Matched Sugeno Iteration Limits',
    'Sequential Results When Adding the Original Branch C Components in Order',
    'Sequential Results After Removing One Original Branch C Component',
    'Final Holdout Results After Removing One Original Branch C Component',
    'Fusion Results Before and After Joint Removal of P, E, and U',
    'Baseline Comparison Under Matched Evaluation Protocols',
    'Replacement of Individual Branch Mechanisms in Three-Branch Fusion',
    'Matched Comparison of A+LR and Three-Branch Fusion',
    'Cumulative Runtime for Tree Insertion and Rebuilding',
    'Cumulative Runtime for A+LR and Three-Branch Fusion',
    'Results Across Repeated Score-Vector-Grouped Partitions',
    'Joint Versus Coverage Rule Selection at a Matched Iteration Limit',
)
GROUPS = {
    3: ((1, 3, 'Predicted'),),
    **{n: ((1, 3, 'Branch C'), (3, 7, 'Three-branch fusion')) for n in (11, 12, 13)},
    15: ((1, 4, 'Sequential (6 configurations)'), (4, 7, 'Final holdout (6 configurations)'),
         (7, 10, 'Vector-grouped CV (5 folds)')),
    16: ((1, 5, 'Sequential'), (5, 9, 'Final holdout')),
    17: ((2, 5, 'FP-TBF-DI'), (5, 8, 'A+LR')),
    18: ((2, 4, 'Tree maintenance (ms)'), (4, 6, 'Total runtime (s)')),
}
SUBHEADERS = {
    3: {1: 'Malignant', 2: 'Benign'},
    **{n: {1: 'Acc. (%)', 2: 'BA (%)', 3: 'Acc. (%)', 4: 'BA (%)',
           5: 'AUC', 6: 'Net correct'} for n in (11, 12, 13)},
    18: {2: 'Insertion', 3: 'Rebuilding', 4: 'Insertion', 5: 'Rebuilding'},
}
METRIC_COLUMNS = {15: (1, 2, 3), 16: (1, 3), 17: (2, 3)}
METRIC_HEADERS = ('Acc. (%)', 'BA (%)', 'AUC')
NOTES = {
    2: ('Accuracy, precision, sensitivity, specificity, F1, and BA are percentages.',),
    6: ('Changes in correctness are for FP-TBF-DI relative to the listed comparator.',),
    14: ('Accuracy and AUC refer to the final fusion. C net correct is reduced minus original for branch C alone.',),
    **{n: ('Net correct is the change in correct fusion predictions relative to the full five-component model.',)
       for n in (11, 12, 13)},
    16: ('Net correct is the change in correct predictions relative to FP-TBF-DI.',),
    17: ('Net correct is the number of correct predictions with A+LR minus that with FP-TBF-DI.',),
    21: ('Differences are joint selection minus coverage selection.',
         'Grouped net counts are averaged across six partitions. Other net counts are totals.'),
}


def cells(line):
    return [v.strip() for v in line.strip().strip('|').split('|')]


def english(value):
    translated = LABELS.get(value, value).replace('、', ', ').replace('～', '–')
    if re.search(r'[\u3400-\u9fff]', translated):
        raise ValueError(f'Missing English label for {value!r}')
    original_numbers = re.findall(r'\d+', value)
    if original_numbers:
        assert original_numbers == re.findall(r'\d+', translated), value
    return translated


def layout_cells(number, table):
    lines = table.strip().splitlines()
    headers, rules = map(cells, lines[:2])
    headers = list(map(english, headers))
    rows = [list(map(english, cells(line))) for line in lines[2:]]
    for j in reversed(METRIC_COLUMNS.get(number, ())):
        headers[j:j+1] = METRIC_HEADERS
        rules[j:j+1] = [':---:'] * len(METRIC_HEADERS)
        for row in rows:
            values = row[j].split(' / ')
            assert len(values) == len(METRIC_HEADERS)
            row[j:j+1] = values
    if number == 16:
        headers[4] = headers[8] = 'Net correct'
    if number == 17:
        headers[8] = 'Net correct'
    for j, label in SUBHEADERS.get(number, {}).items():
        headers[j] = label
    assert all(len(row) == len(headers) == len(rules) for row in rows)
    return headers, rules, rows


def render(number, table, file):
    headers, rules, rows = layout_cells(number, table)
    fonts = Path('C:/Windows/Fonts')
    regular = ImageFont.truetype(str(fonts / 'times.ttf'), 43)
    groups = GROUPS.get(number, ())
    margin, padding, height = 22, 19, 54
    widths = [round(max(regular.getlength(label),
                        *(regular.getlength(row[j]) for row in rows))) + padding*2
              for j, label in enumerate(headers)]
    for start, end, title in groups:
        deficit = round(regular.getlength(title)) + 2*padding - sum(widths[start:end])
        if deficit > 0:
            widths[start] += deficit
    total = sum(widths)
    offsets = [sum(widths[:j]) for j in range(len(widths)+1)]
    top = margin
    header_top = top + (height if groups else 0)
    bottom = header_top + (len(rows)+1)*height
    canvas = Image.new('RGB', (total+2*margin, bottom+margin), 'white')
    draw = ImageDraw.Draw(canvas)
    ink = 'black'
    for start, end, title in groups:
        left, right = (margin+offsets[j] for j in (start, end))
        draw.text(((left+right)/2, top+height/2), title, font=regular, fill=ink, anchor='mm')
        draw.line((left+6, header_top, right-6, header_top), fill=ink, width=2)
    for i, row in enumerate([headers, *rows]):
        y = header_top+i*height
        x = margin
        for j, (value, width) in enumerate(zip(row, widths, strict=True)):
            centered = rules[j].endswith(':')
            centered_y = y + height/2
            if i == 0 and groups and not any(a <= j < b for a,b,_ in groups):
                centered_y -= height/2
            draw.text((x+width/2 if centered else x+padding, centered_y), value,
                      font=regular, fill=ink, anchor='mm' if centered else 'lm')
            x += width
    for y, weight in [(top,4),(header_top+height,3),(bottom,4)]:
        draw.line((margin,y,margin+total,y), fill=ink, width=weight)
    canvas.save(file, dpi=(300,300), optimize=True)


def main():
    source = MANUSCRIPT.read_text(encoding='utf-8')
    matches = list(PATTERN.finditer(source))
    if not matches:
        tables = json.loads(TABLE_SOURCE.read_text(encoding='utf-8'))
        assert list(tables) == list(map(str, range(1, 22)))
        for key, table in tables.items():
            assert f'figures/submission_tables/table_{int(key):02d}.png' in source
            render(int(key), table, ASSETS / f'table_{int(key):02d}.png')
        return
    assert [int(m['number']) for m in matches] == list(range(1,22))
    ASSETS.mkdir(parents=True, exist_ok=True)

    def replace(match):
        number = int(match['number'])
        name = f'table_{number:02d}.png'
        render(number, match['table'], ASSETS / name)
        caption = match['caption']
        return (f'{caption}\n\n![表{number}](figures/submission_tables/{name})\n\n'
                f'<details>\n<summary>查看表格数据</summary>\n\n'
                f'{match["table"]}\n</details>\n')

    updated = PATTERN.sub(replace, source)
    assert [m['table'] for m in PATTERN.finditer(updated)] == [m['table'] for m in matches]
    MANUSCRIPT.write_text(updated, encoding='utf-8')


if __name__ == '__main__':
    main()
