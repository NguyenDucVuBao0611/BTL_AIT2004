import http.server
import socketserver
import json
import os
import sys

PORT = 8000

# Xác định đường dẫn file JSON kết quả
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(BASE_DIR, "results", "ai_thinking_state.json")

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Tắt log yêu cầu HTTP trên màn hình terminal để sạch sẽ
        pass

    def do_GET(self):
        if self.path == "/api/thinking":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            
            data = {"status": "waiting", "message": "Chưa có lượt chơi nào từ AI được ghi nhận."}
            if os.path.exists(JSON_PATH):
                try:
                    with open(JSON_PATH, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    # Tránh lỗi khi Pygame đang mở/ghi đè file
                    data = {"status": "updating", "message": "Đang cập nhật trạng thái..."}
            
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
            
        elif self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
        else:
            # Các tài nguyên khác trả về 404 để bảo mật
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 Not Found")

HTML_CONTENT = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Brain Monitor - Liar's Dice</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0b0f19;
            --surface-glass: rgba(17, 25, 40, 0.65);
            --border-glass: rgba(255, 255, 255, 0.08);
            --primary: #9d4edd;
            --primary-glow: rgba(157, 78, 221, 0.4);
            --accent-cyan: #00f5d4;
            --accent-cyan-glow: rgba(0, 245, 212, 0.3);
            --accent-emerald: #10b981;
            --accent-emerald-glow: rgba(16, 185, 129, 0.3);
            --accent-rose: #f43f5e;
            --accent-rose-glow: rgba(244, 63, 94, 0.3);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --card-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg);
            background-image: 
                radial-gradient(at 10% 10%, rgba(157, 78, 221, 0.15) 0px, transparent 50%),
                radial-gradient(at 90% 90%, rgba(0, 245, 212, 0.1) 0px, transparent 50%);
            color: var(--text-main);
            font-family: 'Outfit', sans-serif;
            min-height: 100vh;
            padding: 2.5rem 2rem;
            display: flex;
            flex-direction: column;
            gap: 2rem;
            overflow-x: hidden;
        }

        /* Glassmorphic Cards */
        .glass-card {
            background: var(--surface-glass);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-glass);
            border-radius: 24px;
            box-shadow: var(--card-shadow);
            padding: 2rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .glass-card:hover {
            border-color: rgba(255, 255, 255, 0.15);
            box-shadow: 0 12px 40px 0 rgba(157, 78, 221, 0.12);
        }

        /* Layout */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 1.5rem;
        }

        .header-title h1 {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff 40%, var(--primary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -1px;
        }

        .header-title p {
            color: var(--text-muted);
            font-size: 1rem;
            margin-top: 0.25rem;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-glass);
            padding: 0.6rem 1.2rem;
            border-radius: 50px;
            font-size: 0.9rem;
            font-weight: 500;
        }

        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background-color: var(--accent-emerald);
            box-shadow: 0 0 12px var(--accent-emerald);
            animation: pulse 2s infinite;
        }

        .status-dot.waiting {
            background-color: #fbbf24;
            box-shadow: 0 0 12px #fbbf24;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); opacity: 0.5; }
            50% { transform: scale(1.05); opacity: 1; }
            100% { transform: scale(0.95); opacity: 0.5; }
        }

        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 1.3fr;
            gap: 2rem;
        }

        @media (max-width: 1024px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }

        .column {
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }

        h2.card-title {
            font-size: 1.35rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.6rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            padding-bottom: 0.75rem;
            letter-spacing: -0.2px;
        }

        h2.card-title svg {
            width: 22px;
            height: 22px;
            color: var(--primary);
        }

        /* Widgets */
        .info-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.25rem;
        }

        .info-row:last-child {
            margin-bottom: 0;
        }

        .info-label {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .info-value {
            font-weight: 600;
            font-size: 1.1rem;
        }

        .info-value.highlight {
            color: var(--primary);
            text-shadow: 0 0 8px var(--primary-glow);
        }

        /* Dice Display */
        .die {
            display: inline-flex;
            width: 46px;
            height: 46px;
            background: #ffffff;
            border-radius: 10px;
            border: 2px solid #e2e8f0;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 1.5rem;
            color: #0f172a;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        .die.wild {
            background: linear-gradient(135deg, #fef08a, #facc15);
            border-color: #eab308;
            color: #713f12;
            position: relative;
        }

        .die.wild::after {
            content: "★";
            font-size: 0.75rem;
            position: absolute;
            top: 2px;
            right: 4px;
            color: #ca8a04;
        }

        .dice-container {
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin-top: 0.5rem;
        }

        /* Bayesian suspicial threshold visualizer */
        .gauge-container {
            position: relative;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 1.75rem;
            margin-top: 1.5rem;
            margin-bottom: 1.5rem;
        }

        .gauge-labels {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-bottom: 0.75rem;
            font-weight: 500;
        }

        .gauge-track {
            height: 14px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 7px;
            position: relative;
            overflow: visible;
        }

        .gauge-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-emerald), var(--primary), var(--accent-rose));
            border-radius: 7px;
            width: 0%;
            transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .gauge-marker {
            position: absolute;
            top: -6px;
            width: 4px;
            height: 26px;
            background: #ffffff;
            box-shadow: 0 0 12px #ffffff;
            border-radius: 2px;
            transition: left 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            z-index: 2;
        }

        .gauge-marker-label {
            position: absolute;
            top: -24px;
            transform: translateX(-50%);
            font-size: 0.75rem;
            font-weight: 700;
            color: #ffffff;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 2px 6px;
            border-radius: 4px;
            white-space: nowrap;
        }

        .gauge-value-label {
            position: absolute;
            top: 22px;
            transform: translateX(-50%);
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--accent-cyan);
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(0, 245, 212, 0.2);
            padding: 2px 6px;
            border-radius: 4px;
            white-space: nowrap;
            transition: left 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* CFR Bars */
        .cfr-item {
            margin-bottom: 1.25rem;
        }

        .cfr-item:last-child {
            margin-bottom: 0;
        }

        .cfr-item-meta {
            display: flex;
            justify-content: space-between;
            font-size: 0.95rem;
            margin-bottom: 0.4rem;
        }

        .cfr-bar-container {
            height: 10px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 5px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.02);
        }

        .cfr-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary), var(--accent-cyan));
            border-radius: 5px;
            width: 0%;
            transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* Tables styling */
        table.profile-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }

        table.profile-table th {
            text-align: left;
            color: var(--text-muted);
            font-weight: 600;
            padding: 1rem 0.75rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.5px;
        }

        table.profile-table td {
            padding: 0.9rem 0.75rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        }

        table.profile-table tr:last-child td {
            border-bottom: none;
        }

        table.profile-table tr:hover td {
            background: rgba(255, 255, 255, 0.015);
        }

        .rate-badge {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 700;
            text-align: center;
        }

        .rate-low {
            background: rgba(16, 185, 129, 0.12);
            color: var(--accent-emerald);
            border: 1px solid rgba(16, 185, 129, 0.2);
        }

        .rate-medium {
            background: rgba(245, 158, 11, 0.12);
            color: #f59e0b;
            border: 1px solid rgba(245, 158, 11, 0.2);
        }

        .rate-high {
            background: rgba(244, 63, 94, 0.12);
            color: var(--accent-rose);
            border: 1px solid rgba(244, 63, 94, 0.2);
        }

        .mono {
            font-family: 'JetBrains Mono', monospace;
        }

        /* Decisions Box alerts */
        .decision-alert {
            display: flex;
            align-items: center;
            gap: 0.85rem;
            padding: 1.25rem;
            border-radius: 16px;
            margin-top: 1.5rem;
            border: 1px solid transparent;
            font-size: 0.95rem;
            font-weight: 500;
            line-height: 1.4;
        }

        .decision-alert.challenge {
            background: rgba(244, 63, 94, 0.06);
            border-color: rgba(244, 63, 94, 0.18);
            color: #fda4af;
        }

        .decision-alert.bid {
            background: rgba(16, 185, 129, 0.06);
            border-color: rgba(16, 185, 129, 0.18);
            color: #a7f3d0;
        }

        .decision-alert svg {
            width: 22px;
            height: 22px;
            flex-shrink: 0;
        }

        .placeholder-box {
            text-align: center;
            padding: 4rem 2rem;
            color: var(--text-muted);
            font-style: italic;
        }

        .placeholder-box svg {
            width: 48px;
            height: 48px;
            margin-bottom: 1rem;
            opacity: 0.3;
            color: var(--text-muted);
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-title">
            <h1>AI Brain Monitor</h1>
            <p>Bảng theo dõi luồng tư duy, xác suất và cấu trúc chiến thuật của AI Liar's Dice thời gian thực</p>
        </div>
        <div class="status-badge">
            <span class="status-dot" id="status-dot"></span>
            <span id="status-text">Đang kết nối game...</span>
        </div>
    </div>

    <div class="dashboard-grid">
        <!-- CỘT 1 -->
        <div class="column">
            <!-- Trạng thái trận đấu -->
            <div class="glass-card">
                <h2 class="card-title">
                    <svg fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4"></path>
                    </svg>
                    Trạng Thái Trận Đấu
                </h2>
                <div class="info-row">
                    <span class="info-label">Giai đoạn hiện tại:</span>
                    <span class="info-value highlight" id="game-phase">-</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Chuỗi trận thích nghi:</span>
                    <span class="info-value" id="player-streak">-</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Số xúc xắc còn lại của bạn:</span>
                    <span class="info-value" id="player-dice-count">-</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Nước cược hiện có trên bàn:</span>
                    <span class="info-value highlight" id="current-bid">-</span>
                </div>
                <div class="info-row" style="flex-direction: column; align-items: flex-start; gap: 0.5rem; margin-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.05); padding-top: 1rem;">
                    <span class="info-label">Xúc xắc của AI (được phép biết):</span>
                    <div class="dice-container" id="ai-dice-container"></div>
                </div>
            </div>

            <!-- Quyết định của AI -->
            <div class="glass-card">
                <h2 class="card-title">
                    <svg fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path>
                    </svg>
                    Quyết Định & Logic AI
                </h2>
                <div class="info-row">
                    <span class="info-label">Nước đi vừa chọn:</span>
                    <span class="info-value" id="ai-action-chosen" style="color: var(--accent-cyan); text-shadow: 0 0 8px var(--accent-cyan-glow); font-size: 1.25rem;">-</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Chế độ tư duy hoạt động:</span>
                    <span class="info-value highlight" id="ai-mode">-</span>
                </div>
                <div class="info-row" id="cfr-visits-row" style="display: none;">
                    <span class="info-label">Số lượt ghé thăm Infoset này:</span>
                    <span class="info-value mono" id="cfr-visits">-</span>
                </div>

                <!-- Bayesian Visualization Panel (khi ở Bayes / Fallback) -->
                <div id="bayesian-decision-panel" style="display: none;">
                    <div class="gauge-container">
                        <div class="gauge-labels">
                            <span>Thật Thà (P=1.0)</span>
                            <span>Gian Lận (P=0.0)</span>
                        </div>
                        <div class="gauge-track">
                            <div class="gauge-fill" id="gauge-p-truth-fill"></div>
                            <div class="gauge-marker" id="gauge-threshold-marker">
                                <div class="gauge-marker-label">Ngưỡng nghi ngờ</div>
                            </div>
                            <div class="gauge-value-label" id="gauge-p-truth-label">P_truth</div>
                        </div>
                    </div>
                    
                    <div class="info-row" style="margin-top: 2.25rem;">
                        <span class="info-label">Xác suất đối thủ nói thật (P_truth):</span>
                        <span class="info-value mono" id="stat-p-truth">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Ngưỡng nghi ngờ động (Threshold):</span>
                        <span class="info-value mono" id="stat-threshold">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Hệ số bước nhảy cược (Modifier):</span>
                        <span class="info-value mono" id="stat-modifier">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Bluff rate trong ngữ cảnh (Base Rate):</span>
                        <span class="info-value mono" id="stat-base-rate">-</span>
                    </div>

                    <!-- Alert message -->
                    <div class="decision-alert" id="decision-alert-box">
                        <svg id="alert-icon" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"></svg>
                        <span id="decision-alert-text">-</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- CỘT 2 -->
        <div class="column">
            <!-- CFR Strategy -->
            <div class="glass-card" id="cfr-distribution-card">
                <h2 class="card-title">
                    <svg fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M4 6h16M4 12h16M4 18h7"></path>
                    </svg>
                    Phân Phối Chiến Thuật CFR (Strategy Distribution)
                </h2>
                <div id="cfr-no-data" class="placeholder-box">
                    <svg fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path>
                    </svg>
                    <div>AI đang ứng biến Bayes hoặc infoset chưa đủ tin cậy. Bảng chiến lược CFR chưa kích hoạt.</div>
                </div>
                <div id="cfr-items-container"></div>
            </div>

            <!-- Hồ sơ thói quen người chơi (Bayesian context table) -->
            <div class="glass-card" id="bayesian-profile-card">
                <h2 class="card-title">
                    <svg fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
                    </svg>
                    Hồ Sơ Thói Quen Của Bạn (Bayes Multi-Context)
                </h2>
                <div style="overflow-x: auto;">
                    <table class="profile-table">
                        <thead>
                            <tr>
                                <th>Ngữ Cảnh Học</th>
                                <th style="text-align: center; width: 100px;">Lượt Bị Tố</th>
                                <th style="text-align: center; width: 80px;">Nói Dối</th>
                                <th style="text-align: right; width: 120px;">Tỷ Lệ Bluff</th>
                            </tr>
                        </thead>
                        <tbody id="profile-table-body">
                            <!-- JS fill -->
                        </tbody>
                    </table>
                </div>
                <div class="info-row" style="margin-top: 1.5rem; border-top: 1px solid rgba(255, 255, 255, 0.05); padding-top: 1.25rem;">
                    <span class="info-label">Cược mặt lớn (>= 4) của bạn:</span>
                    <span class="info-value mono" id="our-bids-high">-</span>
                </div>
                <div class="info-row">
                    <span class="info-label">AI thách thức cược lớn:</span>
                    <span class="info-value mono" id="opp-challenges-high">-</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Tỷ lệ AI phát hiện cược lớn (Exploit):</span>
                    <span class="info-value highlight" id="exploit-rate" style="color: var(--accent-cyan); text-shadow: 0 0 8px var(--accent-cyan-glow);">-</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        let lastTimestamp = 0;

        function formatBid(bid) {
            if (!bid) return 'Chưa có cược';
            return `${bid.quantity} con mặt [${bid.face_value}]`;
        }

        function getBluffBadgeClass(rate) {
            if (rate < 0.26) return 'rate-badge rate-low';
            if (rate < 0.46) return 'rate-badge rate-medium';
            return 'rate-badge rate-high';
        }

        async function fetchThinkingState() {
            try {
                const response = await fetch('/api/thinking');
                if (!response.ok) throw new Error('Không thể tải API');
                const data = await response.json();
                
                // Active status
                document.getElementById('status-dot').className = 'status-dot';
                document.getElementById('status-text').innerText = 'Kết nối hoạt động';

                if (data.timestamp !== lastTimestamp) {
                    lastTimestamp = data.timestamp;
                    updateDashboard(data);
                }
            } catch (error) {
                console.error(error);
                document.getElementById('status-dot').className = 'status-dot waiting';
                document.getElementById('status-text').innerText = 'Đang chờ game khởi chạy...';
            }
        }

        function updateDashboard(data) {
            // General Info
            document.getElementById('game-phase').innerText = data.phase || 'Đang chơi';
            
            let streakText = 'Cân bằng (0)';
            if (data.player_streak > 0) {
                streakText = `🔥 Bạn thắng liên tiếp ${data.player_streak} ván`;
            } else if (data.player_streak < 0) {
                streakText = `❄️ AI dẫn trước ${-data.player_streak} ván`;
            }
            document.getElementById('player-streak').innerText = streakText;

            // Opponent Dice
            document.getElementById('player-dice-count').innerText = `${data.opponent_dice_count} xúc xắc`;

            // Current Bid
            document.getElementById('current-bid').innerText = formatBid(data.current_bid);

            // Action Chosen
            document.getElementById('ai-action-chosen').innerText = data.action_chosen || 'Đang tính toán...';

            // Mode
            document.getElementById('ai-mode').innerText = data.mode || '-';

            // AI dice
            const diceContainer = document.getElementById('ai-dice-container');
            diceContainer.innerHTML = '';
            if (data.ai_dice && data.ai_dice.length > 0) {
                data.ai_dice.forEach(value => {
                    const die = document.createElement('div');
                    die.className = value === 1 ? 'die wild' : 'die';
                    die.innerText = value;
                    diceContainer.appendChild(die);
                });
            } else {
                diceContainer.innerText = 'Ẩn bài';
            }

            // Mode logic: CFR vs Bayes
            if (data.mode === 'CFR (Tra bảng)') {
                document.getElementById('cfr-visits-row').style.display = 'flex';
                document.getElementById('cfr-visits').innerText = data.visits || 0;
                document.getElementById('cfr-no-data').style.display = 'none';
                
                const cfrContainer = document.getElementById('cfr-items-container');
                cfrContainer.innerHTML = '';
                
                const dist = data.distribution || {};
                const sortedActions = Object.entries(dist).sort((a, b) => b[1] - a[1]);
                
                if (sortedActions.length > 0) {
                    sortedActions.forEach(([actionName, probability]) => {
                        const item = document.createElement('div');
                        item.className = 'cfr-item';
                        
                        let displayAction = actionName;
                        if (actionName === 'Challenge') {
                            displayAction = 'Challenge (Tố cáo)';
                        } else if (actionName.startsWith('Bid:')) {
                            const parts = actionName.split(':');
                            displayAction = `Cược ${parts[1]} con mặt [${parts[2]}]`;
                        }
                        
                        const pct = (probability * 100).toFixed(1);
                        item.innerHTML = `
                            <div class="cfr-item-meta">
                                <span>${displayAction}</span>
                                <span class="mono">${pct}%</span>
                            </div>
                            <div class="cfr-bar-container">
                                <div class="cfr-bar-fill" style="width: ${pct}%"></div>
                            </div>
                        `;
                        cfrContainer.appendChild(item);
                    });
                } else {
                    cfrContainer.innerText = 'Không tìm thấy phân phối chiến lược CFR.';
                }

                document.getElementById('bayesian-decision-panel').style.display = 'none';
            } else {
                // Bayes
                document.getElementById('cfr-visits-row').style.display = 'none';
                document.getElementById('cfr-no-data').style.display = 'block';
                document.getElementById('cfr-items-container').innerHTML = '';
                document.getElementById('bayesian-decision-panel').style.display = 'block';

                const p_truth = data.p_truth !== undefined && data.p_truth !== null ? data.p_truth : 0.5;
                const threshold = data.threshold !== undefined && data.threshold !== null ? data.threshold : 0.5;
                const modifier = data.modifier !== undefined && data.modifier !== null ? data.modifier : 1.0;
                const base_rate = data.base_rate !== undefined && data.base_rate !== null ? data.base_rate : 0.33;

                document.getElementById('stat-p-truth').innerText = p_truth.toFixed(4);
                document.getElementById('stat-threshold').innerText = threshold.toFixed(4);
                document.getElementById('stat-modifier').innerText = modifier.toFixed(2);
                document.getElementById('stat-base-rate').innerText = base_rate.toFixed(4);

                // Progress gauge
                // P_truth = 1.0 (Thật thà) -> left (0% nghi ngờ)
                // P_truth = 0.0 (Láo) -> right (100% nghi ngờ)
                const suspicionPercent = (1.0 - p_truth) * 100;
                const thresholdPercent = (1.0 - threshold) * 100;

                document.getElementById('gauge-p-truth-fill').style.width = `${suspicionPercent}%`;
                document.getElementById('gauge-threshold-marker').style.left = `${thresholdPercent}%`;
                
                const valLabel = document.getElementById('gauge-p-truth-label');
                valLabel.style.left = `${suspicionPercent}%`;
                valLabel.innerText = `Độ nghi ngờ = ${suspicionPercent.toFixed(1)}%`;

                // Alert action recommendation
                const alertBox = document.getElementById('decision-alert-box');
                const alertIcon = document.getElementById('alert-icon');
                const alertText = document.getElementById('decision-alert-text');
                
                if (p_truth < threshold) {
                    alertBox.className = 'decision-alert challenge';
                    alertIcon.innerHTML = `
                        <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
                    `;
                    alertText.innerText = `AI QUYẾT ĐỊNH: TỐ CÁO (CHALLENGE) vì Độ nghi ngờ (${suspicionPercent.toFixed(1)}%) vượt quá Ngưỡng tối đa cho phép (${thresholdPercent.toFixed(1)}%).`;
                } else {
                    alertBox.className = 'decision-alert bid';
                    alertIcon.innerHTML = `
                        <path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    `;
                    alertText.innerText = `AI QUYẾT ĐỊNH: TIN TƯỞNG & CƯỢC TIẾP vì Độ nghi ngờ (${suspicionPercent.toFixed(1)}%) vẫn nằm trong Ngưỡng an toàn (${thresholdPercent.toFixed(1)}%).`;
                }
            }

            // Bayesian profiling
            const bm = data.bayesian_context_metrics;
            const profileBody = document.getElementById('profile-table-body');
            profileBody.innerHTML = '';
            
            if (bm) {
                const contexts = [
                    { name: 'Mặc định (General)', resolved: bm.resolved_opp_bids, bluffs: bm.opp_bluffs },
                    { name: 'Bạn ít xúc xắc (<= 2)', resolved: bm.resolved_opp_bids_low_dice, bluffs: bm.opp_bluffs_low_dice },
                    { name: 'Thế trận: Bạn đang thắng', resolved: bm.resolved_opp_bids_winning, bluffs: bm.opp_bluffs_winning },
                    { name: 'Thế trận: Bạn đang thua', resolved: bm.resolved_opp_bids_losing, bluffs: bm.opp_bluffs_losing },
                    { name: 'Bạn cược số lượng lớn (>= 50%)', resolved: bm.resolved_opp_bids_high_qty, bluffs: bm.opp_bluffs_high_qty },
                    { name: 'Bạn cược xúc xắc mặt [1]', resolved: bm.resolved_opp_bids_face1, bluffs: bm.opp_bluffs_face1 }
                ];

                contexts.forEach(ctx => {
                    const alpha = 1.0;
                    const beta = 2.0;
                    const estimatedRate = (ctx.bluffs + alpha) / (ctx.resolved + alpha + beta);
                    const pct = (estimatedRate * 100).toFixed(1);
                    
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td><strong>${ctx.name}</strong></td>
                        <td style="text-align: center;" class="mono">${ctx.resolved}</td>
                        <td style="text-align: center;" class="mono">${ctx.bluffs}</td>
                        <td style="text-align: right;"><span class="${getBluffBadgeClass(estimatedRate)}">${pct}%</span></td>
                    `;
                    profileBody.appendChild(row);
                });

                document.getElementById('our-bids-high').innerText = bm.our_bids_high_face || 0;
                document.getElementById('opp-challenges-high').innerText = bm.opp_challenges_high_face || 0;
                
                if (bm.our_bids_high_face > 0) {
                    const rate = (bm.opp_challenges_high_face / bm.our_bids_high_face * 100).toFixed(1);
                    document.getElementById('exploit-rate').innerText = `${rate}% (AI trừng phạt cược láo mặt lớn)`;
                } else {
                    document.getElementById('exploit-rate').innerText = 'Chưa có cược mặt >= 4';
                }
            } else {
                profileBody.innerHTML = `
                    <tr>
                        <td colspan="4" style="text-align: center; color: var(--text-muted); padding: 2rem;">
                            Đang đợi dữ liệu Bayesian cập nhật...
                        </td>
                    </tr>
                `;
                document.getElementById('our-bids-high').innerText = '-';
                document.getElementById('opp-challenges-high').innerText = '-';
                document.getElementById('exploit-rate').innerText = '-';
            }
        }

        fetchThinkingState();
        setInterval(fetchThinkingState, 500);
    </script>
</body>
</html>
"""

def run():
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
            print("======================================================================")
            print(f"🚀 AI Thinking Monitor Dashboard running at http://localhost:{PORT}")
            print("Vui lòng mở liên kết trên bằng trình duyệt để theo dõi suy nghĩ của AI.")
            print("Nhấn Ctrl + C để tắt Dashboard Server.")
            print("======================================================================")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nĐang dừng Dashboard Server...")
        sys.exit(0)
    except Exception as e:
        print(f"Lỗi khởi chạy server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run()
