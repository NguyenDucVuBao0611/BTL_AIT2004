import pytest
from core.probability import binomial_probability

def test_binomial_probability_boundaries():
    # Trường hợp k <= 0 phải luôn trả về 1.0
    assert binomial_probability(0, 5, 1/3) == 1.0
    assert binomial_probability(-1, 5, 1/3) == 1.0
    
    # Trường hợp k > n phải luôn trả về 0.0
    assert binomial_probability(6, 5, 1/3) == 0.0
    assert binomial_probability(10, 2, 1/6) == 0.0
    
    # Trường hợp số lượng ẩn n < 0
    assert binomial_probability(1, -1, 0.5) == 0.0

    # Trường hợp xác suất p ngoài biên
    assert binomial_probability(2, 5, 0.0) == 0.0
    assert binomial_probability(2, 5, 1.0) == 1.0

def test_binomial_probability_exact_values():
    # Thử nghiệm với các giá trị tính toán tay chuẩn xác
    
    # 1. n = 1, p = 0.5, k = 1: P(X >= 1) = 0.5
    assert abs(binomial_probability(1, 1, 0.5) - 0.5) < 1e-9

    # 2. n = 2, p = 0.5, k = 2: P(X >= 2) = P(X = 2) = 0.25
    assert abs(binomial_probability(2, 2, 0.5) - 0.25) < 1e-9

    # 3. n = 3, p = 1/3, k = 1: P(X >= 1) = 1 - P(X = 0) = 1 - (2/3)^3 = 19/27 ~= 0.7037037037
    assert abs(binomial_probability(1, 3, 1/3) - 19/27) < 1e-9

    # 4. n = 3, p = 1/3, k = 2: P(X >= 2) = C(3,2)*(1/3)^2*(2/3) + C(3,3)*(1/3)^3 = 6/27 + 1/27 = 7/27 ~= 0.259259259
    assert abs(binomial_probability(2, 3, 1/3) - 7/27) < 1e-9
