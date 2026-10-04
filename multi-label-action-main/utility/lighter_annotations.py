import os
import csv
import argparse

def create_lighter_annots(ipfile, opfile):
    if not os.path.exists(ipfile):
        raise FileNotFoundError(f"Input annotation file not found: {ipfile}")

    os.makedirs(os.path.dirname(os.path.abspath(opfile)), exist_ok=True)
    ovids = set()
    rows = []
    with open(ipfile, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=' ')
        for row in reader:
            ovid = row.get('original_vido_id') or row.get('video_id')
            labels = row.get('labels', '')
            if not labels or not ovid or ovid in ovids:
                continue
            ovids.add(ovid)
            rows.append([ovid, labels])

    header = ['video_id', 'labels']
    with open(opfile, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    print(f"[INFO] Processed {len(rows)} video annotations. Saved to: {opfile}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate lighter annotation CSVs for Animal Kingdom")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input train.csv or val.csv")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output train_light.csv or val_light.csv")
    args = parser.parse_args()

    create_lighter_annots(args.input, args.output)