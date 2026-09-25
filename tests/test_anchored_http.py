"""固定观测校正的 HTTP 层测试：422 校验契约、409 冲突契约、
响应结构与端到端精确数值，以及冲突后服务仍可正常调用。"""

from __future__ import annotations

from fractions import Fraction

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def obs(id_, reading, weight=1, **extra):
    item = {"id": id_, "reading": reading, "weight": weight}
    item.update(extra)
    return item


def anchored_payload(items, anchors, **extra):
    payload = {"observations": items, "anchors": anchors}
    payload.update(extra)
    return payload


def _fraction(model):
    return Fraction(model["numerator"], model["denominator"])


# ---------------------------------------------------------------------------
# 正常路径
# ---------------------------------------------------------------------------


def test_anchored_basic_response_marks_fixed_points():
    payload = anchored_payload(
        [obs("a", 3), obs("b", 1), obs("c", 2)], ["c"]
    )
    response = client.post("/correct/anchored", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()

    # 锚点 c 原值保留；a,b 被上界 2 压平
    assert data["blocks"] == [
        {"start_index": 0, "end_index": 2, "mean": {"numerator": 2, "denominator": 1}}
    ]
    assert data["fitted"] == [
        {"id": "a", "fitted": {"numerator": 2, "denominator": 1}, "fixed": False},
        {"id": "b", "fitted": {"numerator": 2, "denominator": 1}, "fixed": False},
        {"id": "c", "fitted": {"numerator": 2, "denominator": 1}, "fixed": True},
    ]
    assert data["total_error"] == {"numerator": 2, "denominator": 1}


def test_anchored_fitted_equals_reading_at_anchor_exactly():
    # 人工复测点原值保留：锚点读数不被平滑曲线改写
    payload = anchored_payload(
        [obs("p1", 0, 5), obs("p2", 0, 5), obs("p3", 7, 1), obs("p4", 0, 5), obs("p5", 0, 5)],
        ["p3"],
    )
    response = client.post("/correct/anchored", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    by_id = {p["id"]: p for p in data["fitted"]}
    assert by_id["p3"]["fixed"] is True
    assert _fraction(by_id["p3"]["fitted"]) == 7
    # 右段被下界 7 托起，左段保持 0
    assert [_fraction(p["fitted"]) for p in data["fitted"]] == [
        Fraction(0), Fraction(0), Fraction(7), Fraction(7), Fraction(7)
    ]


def test_anchored_equal_values_merge_across_anchors_into_one_block():
    payload = anchored_payload(
        [obs("a", 3), obs("b", 1), obs("c", 3), obs("d", 1), obs("e", 3)],
        ["b", "d"],
    )
    response = client.post("/correct/anchored", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    # 等值跨锚点也只能作为一个分块：(0..3) 全为 1
    assert data["blocks"] == [
        {"start_index": 0, "end_index": 3, "mean": {"numerator": 1, "denominator": 1}},
        {"start_index": 4, "end_index": 4, "mean": {"numerator": 3, "denominator": 1}},
    ]
    assert data["total_error"] == {"numerator": 8, "denominator": 1}
    assert [p["fixed"] for p in data["fitted"]] == [False, True, False, True, False]


def test_anchored_endpoint_anchors():
    # 首点锚定：其余点受其读数下界约束
    payload = anchored_payload([obs("a", 2), obs("b", 0), obs("c", 1)], ["a"])
    response = client.post("/correct/anchored", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert [_fraction(p["fitted"]) for p in data["fitted"]] == [Fraction(2)] * 3

    # 末点锚定：其余点受其读数上界约束
    payload = anchored_payload([obs("a", 1), obs("b", 9), obs("c", 2)], ["c"])
    response = client.post("/correct/anchored", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert [_fraction(p["fitted"]) for p in data["fitted"]] == [
        Fraction(1), Fraction(2), Fraction(2)
    ]
    # 跨锚点等值合并：b 与锚点 c 同处一块
    assert data["blocks"][-1] == {
        "start_index": 1,
        "end_index": 2,
        "mean": {"numerator": 2, "denominator": 1},
    }


def test_anchored_fully_fixed_returns_readings_with_zero_error():
    items = [obs("a", 1, 3), obs("b", 1, 1), obs("c", 2, 4), obs("d", 5, 1)]
    payload = anchored_payload(items, ["a", "b", "c", "d"])
    response = client.post("/correct/anchored", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert [_fraction(p["fitted"]) for p in data["fitted"]] == [
        Fraction(1), Fraction(1), Fraction(2), Fraction(5)
    ]
    assert all(p["fixed"] for p in data["fitted"])
    assert data["total_error"] == {"numerator": 0, "denominator": 1}
    # 等读数锚点合并为最大等值块
    assert data["blocks"] == [
        {"start_index": 0, "end_index": 1, "mean": {"numerator": 1, "denominator": 1}},
        {"start_index": 2, "end_index": 2, "mean": {"numerator": 2, "denominator": 1}},
        {"start_index": 3, "end_index": 3, "mean": {"numerator": 5, "denominator": 1}},
    ]


def test_anchored_anchor_id_order_in_request_is_irrelevant():
    items = [obs("a", 1), obs("b", 5), obs("c", 3), obs("d", 9)]
    first = client.post(
        "/correct/anchored", json=anchored_payload(items, ["a", "c"])
    )
    second = client.post(
        "/correct/anchored", json=anchored_payload(items, ["c", "a"])
    )
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()


def test_anchored_max_anchors_boundary_accepted():
    items = [obs(f"id-{i}", i) for i in range(12)]
    anchors = [f"id-{i}" for i in range(12)]
    response = client.post(
        "/correct/anchored", json=anchored_payload(items, anchors)
    )
    assert response.status_code == 200, response.text
    assert all(p["fixed"] for p in response.json()["fitted"])


def test_anchored_end_to_end_matches_solver():
    readings = [10, 4, 8, 2, 9, 3]
    weights = [3, 1, 2, 4, 1, 5]
    anchors = ["p1", "p4"]
    payload = anchored_payload(
        [obs(f"p{i}", readings[i], weights[i]) for i in range(6)], anchors
    )
    response = client.post("/correct/anchored", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()

    from app.isotonic import fit_isotonic_anchored

    result = fit_isotonic_anchored(readings, weights, [1, 4])
    expected = [
        Fraction(num, den)
        for num, den in zip(result.fitted_num, result.fitted_den)
    ]
    got = [_fraction(p["fitted"]) for p in data["fitted"]]
    assert got == expected
    assert [p["fixed"] for p in data["fitted"]] == [
        False, True, False, False, True, False
    ]
    assert _fraction(data["total_error"]) == Fraction(
        result.total_error_num, result.total_error_den
    )
    # 块与逐点曲线一致
    for block in data["blocks"]:
        mean = _fraction(block["mean"])
        for i in range(block["start_index"], block["end_index"] + 1):
            assert got[i] == mean


# ---------------------------------------------------------------------------
# 422：未知/重复锚点、数量越界、类型错误、未知字段（沿用原校验契约）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        # 未知锚点
        anchored_payload([obs("a", 1), obs("b", 2)], ["nope"]),
        anchored_payload([obs("a", 1), obs("b", 2)], ["a", "nope"]),
        # 空 id / 非 ASCII id 作为锚点：必然未知
        anchored_payload([obs("a", 1), obs("b", 2)], [""]),
        anchored_payload([obs("a", 1), obs("b", 2)], ["刀具"]),
        # 重复锚点
        anchored_payload([obs("a", 1), obs("b", 2)], ["a", "a"]),
        anchored_payload([obs("a", 1), obs("b", 2), obs("c", 3)], ["a", "c", "a"]),
        # 锚点数量越界：0 个 / 13 个
        anchored_payload([obs("a", 1), obs("b", 2)], []),
        anchored_payload(
            [obs(f"id-{i}", i) for i in range(13)],
            [f"id-{i}" for i in range(13)],
        ),
        # 锚点类型错误
        anchored_payload([obs("a", 1), obs("b", 2)], "a"),
        anchored_payload([obs("a", 1), obs("b", 2)], [1]),
        anchored_payload([obs("a", 1), obs("b", 2)], [True]),
        anchored_payload([obs("a", 1), obs("b", 2)], [["a"]]),
        anchored_payload([obs("a", 1), obs("b", 2)], [{"id": "a"}]),
        # 顶层未知字段
        anchored_payload([obs("a", 1), obs("b", 2)], ["a"], mode="aggressive"),
        # 缺 anchors 字段
        {"observations": [obs("a", 1), obs("b", 2)]},
        # 观测本身的校验依旧生效
        anchored_payload([obs("a", 1), obs("a", 2)], ["a"]),
        anchored_payload([obs("a", -1), obs("b", 2)], ["a"]),
        anchored_payload([obs("a", 1, extra=1), obs("b", 2)], ["a"]),
    ],
)
def test_anchored_invalid_payloads_return_422(payload):
    response = client.post("/correct/anchored", json=payload)
    assert response.status_code == 422, response.text


# ---------------------------------------------------------------------------
# 409：锚点读数按加工顺序下降 → INFEASIBLE_ANCHORS，不发布部分拟合
# ---------------------------------------------------------------------------


def test_infeasible_anchors_return_409_with_earliest_conflict():
    # 锚点读数按原顺序 5, 4, 2：最早冲突的相邻锚点对为 (a, c)
    payload = anchored_payload(
        [obs("a", 5), obs("b", 1), obs("c", 4), obs("d", 2)],
        ["a", "c", "d"],
    )
    response = client.post("/correct/anchored", json=payload)
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["error"] == "INFEASIBLE_ANCHORS"
    assert detail["conflicting_anchor_ids"] == ["a", "c"]
    # 不发布部分拟合
    body = response.json()
    assert "blocks" not in body
    assert "fitted" not in body
    assert "total_error" not in body


def test_infeasible_conflict_reports_pair_in_original_order():
    # 锚点数组顺序无关：按观测顺序定位最早冲突对
    payload = anchored_payload(
        [obs("x", 2), obs("y", 2), obs("z", 1)], ["z", "x", "y"]
    )
    response = client.post("/correct/anchored", json=payload)
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["error"] == "INFEASIBLE_ANCHORS"
    assert detail["conflicting_anchor_ids"] == ["y", "z"]


def test_service_remains_usable_after_conflict():
    infeasible = anchored_payload(
        [obs("a", 5), obs("b", 1), obs("c", 4), obs("d", 2)],
        ["a", "c", "d"],
    )
    assert client.post("/correct/anchored", json=infeasible).status_code == 409

    # 冲突之后：原 /correct 不受影响
    plain = client.post(
        "/correct", json={"observations": [obs("a", 3), obs("b", 1), obs("c", 2)]}
    )
    assert plain.status_code == 200, plain.text
    assert plain.json()["total_error"] == {"numerator": 2, "denominator": 1}

    # 修正锚点后的同一批观测也能正常校正（b=1, c=4 非递减，可行）
    fixed = anchored_payload(
        [obs("a", 5), obs("b", 1), obs("c", 4), obs("d", 2)], ["b", "c"]
    )
    response = client.post("/correct/anchored", json=fixed)
    assert response.status_code == 200, response.text
    data = response.json()
    by_id = {p["id"]: p for p in data["fitted"]}
    assert _fraction(by_id["b"]["fitted"]) == 1
    assert _fraction(by_id["c"]["fitted"]) == 4
    # a 被上界 1 压平，d 被下界 4 托起
    assert [_fraction(p["fitted"]) for p in data["fitted"]] == [
        Fraction(1), Fraction(1), Fraction(4), Fraction(4)
    ]

    # 再次冲突也仍然只是 409，服务无状态残留
    assert client.post("/correct/anchored", json=infeasible).status_code == 409


# ---------------------------------------------------------------------------
# 原 /correct 逐项不变；无锚点结果与锚定结果的关系
# ---------------------------------------------------------------------------


def test_correct_response_shape_unchanged_no_fixed_field():
    payload = {"observations": [obs("a", 3), obs("b", 1), obs("c", 2)]}
    response = client.post("/correct", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert set(data.keys()) == {"blocks", "fitted", "total_error"}
    for point in data["fitted"]:
        assert set(point.keys()) == {"id", "fitted"}
    for block in data["blocks"]:
        assert set(block.keys()) == {"start_index", "end_index", "mean"}
    # 数值逐项不变
    assert data["blocks"] == [
        {"start_index": 0, "end_index": 2, "mean": {"numerator": 2, "denominator": 1}}
    ]
    assert data["total_error"] == {"numerator": 2, "denominator": 1}


def test_non_binding_anchor_result_matches_correct_item_by_item():
    # 锚点不约束最优解时，锚定响应与 /correct 逐项一致（除 fixed 标记）
    items = [obs("a", 3), obs("b", 1), obs("c", 2)]
    plain = client.post("/correct", json={"observations": items}).json()
    anchored = client.post(
        "/correct/anchored", json=anchored_payload(items, ["c"])
    ).json()
    assert anchored["blocks"] == plain["blocks"]
    assert anchored["total_error"] == plain["total_error"]
    assert [p["fitted"] for p in anchored["fitted"]] == [
        p["fitted"] for p in plain["fitted"]
    ]
    assert [p["id"] for p in anchored["fitted"]] == [
        p["id"] for p in plain["fitted"]
    ]


def test_correct_still_rejects_anchors_field_as_unknown():
    payload = {
        "observations": [obs("a", 1), obs("b", 2)],
        "anchors": ["a"],
    }
    response = client.post("/correct", json=payload)
    assert response.status_code == 422
