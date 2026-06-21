import math
import functools

@functools.lru_cache(maxsize=None)
def binomial_probability(k: int, n: int, p: float) -> float:
    """Tính xác suất tích lũy P(X >= k) trong phân phối nhị thức B(n, p).
    Được dùng để tính xác suất đối thủ có ít nhất k viên xúc xắc có mặt cược f 
    trong tổng số n viên xúc xắc ẩn của đối thủ.

    Args:
        k (int): Số lượng xúc xắc khớp cần có tối thiểu.
        n (int): Số lượng xúc xắc ẩn (chưa biết) của đối thủ.
        p (float): Xác suất xuất hiện mặt cược trên 1 viên xúc xắc (1/6 hoặc 1/3).

    Returns:
        float: Xác suất tích lũy P(X >= k), nằm trong khoảng [0.0, 1.0].
    """
    # Xử lý các trường hợp biên
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if n < 0:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0

    prob_sum = 0.0
    for i in range(k, n + 1):
        # Tính C(n, i) * p^i * (1-p)^(n-i)
        term = math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        prob_sum += term

    # Giới hạn giá trị trả về trong khoảng [0.0, 1.0] phòng sai số dấu phẩy động
    return min(max(prob_sum, 0.0), 1.0)
