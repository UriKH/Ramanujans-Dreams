from typing import List, Tuple
import csv
import pickle
from ramanujantools import Position
from .models import DataTemplate


def save_csv(data: List[Tuple[Position, float]], filename: str) -> None:
    if not data:
        raise ValueError("No data to save")

    coord_names = sorted(data[0][0].to_dict().keys())

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(coord_names + ["delta"])

        for pos, delta in data:
            d = pos.to_dict()
            row = [d.get(k, 0) for k in coord_names] + [delta]
            writer.writerow(row)


def save_pickle(run: DataTemplate, filename: str) -> None:
    with open(filename + ".pkl", "wb") as f:
        pickle.dump(run, f)
