"""
تست واحد برای توابع مشترک زیرساخت
=====================================================================
هیچ‌کدام از این تست‌ها به دیتاست واقعی Elliptic نیاز ندارند؛ همه روی
داده مصنوعی کوچک اجرا می‌شوند تا سریع و مستقل باشند.

اجرا:
    pip install pytest --if-needed
    pytest test_metrics_utils.py -v
"""

import numpy as np
import torch
import pytest

from metrics_utils import build_edge_index, get_temporal_split_masks, find_best_threshold


# ------------------------------------------------------------
# build_edge_index
# ------------------------------------------------------------

def test_build_edge_index_maps_correctly():
    node_ids = ["a", "b", "c", "d"]
    edge_src = ["a", "b", "c"]
    edge_dst = ["b", "c", "d"]
    map_id, edge_index = build_edge_index(node_ids, edge_src, edge_dst)

    assert map_id == {"a": 0, "b": 1, "c": 2, "d": 3}
    assert edge_index.shape == (2, 3)
    assert edge_index.tolist() == [[0, 1, 2], [1, 2, 3]]


def test_build_edge_index_no_out_of_range_indices():
    node_ids = list(range(100))
    edge_src = [0, 5, 99]
    edge_dst = [1, 6, 0]
    _, edge_index = build_edge_index(node_ids, edge_src, edge_dst)

    assert edge_index.max().item() < len(node_ids)
    assert edge_index.min().item() >= 0


def test_build_edge_index_handles_duplicate_node_order():
    # ترتیب node_ids باید همان ترتیبی باشد که فیچرهای x هم دارند؛
    # این تست مطمئن می‌شود اندیس هر گره دقیقاً برابر جایگاهش در
    # لیست ورودی است، نه ترتیب الفبایی یا هر ترتیب دیگر.
    node_ids = ["z", "a", "m"]
    map_id, _ = build_edge_index(node_ids, [], [])
    assert map_id == {"z": 0, "a": 1, "m": 2}


# ------------------------------------------------------------
# get_temporal_split_masks
# ------------------------------------------------------------

def test_temporal_split_masks_are_mutually_exclusive_and_correct_size():
    time_step = torch.tensor([1, 10, 27, 28, 30, 34, 35, 49])
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    train_mask, val_mask, test_mask = get_temporal_split_masks(
        time_step, y, train_end=27, val_end=34
    )

    overlap = (train_mask & val_mask) | (train_mask & test_mask) | (val_mask & test_mask)
    assert overlap.sum().item() == 0

    assert train_mask.sum().item() == 3   # timesteps 1, 10, 27
    assert val_mask.sum().item() == 3     # timesteps 28, 30, 34
    assert test_mask.sum().item() == 2    # timesteps 35, 49
    assert (train_mask | val_mask | test_mask).sum().item() == len(time_step)


def test_temporal_split_masks_excludes_unlabeled_nodes():
    time_step = torch.tensor([1, 2, 3])
    y = torch.tensor([1, -1, 0])
    train_mask, val_mask, test_mask = get_temporal_split_masks(
        time_step, y, train_end=27, val_end=34
    )
    # گره وسط برچسب -1 دارد و نباید در هیچ‌کدام از سه مجموعه بیفتد
    assert train_mask.tolist() == [True, False, True]
    assert not val_mask.any()
    assert not test_mask.any()


def test_temporal_split_masks_test_set_never_touches_train_boundary():
    time_step = torch.arange(1, 50)
    y = torch.ones(49, dtype=torch.long)
    train_mask, val_mask, test_mask = get_temporal_split_masks(
        time_step, y, train_end=27, val_end=34
    )
    assert time_step[test_mask].min().item() == 35
    assert time_step[train_mask].max().item() == 27


# ------------------------------------------------------------
# find_best_threshold
# ------------------------------------------------------------

def test_find_best_threshold_on_perfectly_separable_case():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    best_t, best_f1 = find_best_threshold(y_true, y_prob)
    assert 0.3 < best_t <= 0.7
    assert best_f1 == pytest.approx(1.0)


def test_find_best_threshold_returns_values_in_valid_range():
    y_true = np.array([0, 1, 0, 1, 0])
    y_prob = np.array([0.2, 0.9, 0.4, 0.6, 0.1])
    best_t, best_f1 = find_best_threshold(y_true, y_prob)
    assert 0.0 <= best_t <= 1.0
    assert 0.0 <= best_f1 <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])