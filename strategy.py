from collections import deque

def bfs_find_tail(start, target, obstacles, width, height):
    """
    Kiểm tra xem có đường từ start (đầu rắn) đến target (đuôi rắn) không.
    obstacles là tập hợp (set) các tọa độ thân rắn (đóng vai trò như vật cản).
    """
    # Nếu đầu ở ngay cạnh đuôi, luôn đúng
    if start == target:
        return True

    queue = deque([start])
    visited = set([start])
    visited.update(obstacles) # Khởi tạo vật cản là các ô đã đi qua để BFS không đi vào

    # 4 hướng: Lên, Xuống, Trái, Phải
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    while queue:
        current = queue.popleft()

        if current == target:
            return True

        # Duyệt 4 hướng xung quanh
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            neighbor = (nx, ny)

            # Kiểm tra xem có nằm trong bàn cờ và chưa bị thăm/không đâm vào thân
            if 0 <= nx < width and 0 <= ny < height:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
    return False # Ngõ cụt


def flood_fill_space(start, obstacles, width=11, height=11):
    """
    Đếm tổng số ô trống liên thông có thể đi tới từ vị trí start.
    """
    queue = deque([start])
    visited = set([start])
    visited.update(obstacles)
    
    space_count = 0
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    while queue:
        current = queue.popleft()
        space_count += 1

        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            neighbor = (nx, ny)

            if 0 <= nx < width and 0 <= ny < height:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
    return space_count

def get_safe_action(head, body, sorted_actions, width, height):
    """
    head: (x, y) - Đầu rắn hiện tại
    body: list of (x, y) - Toàn bộ thân rắn, đuôi là body[-1]
    sorted_actions: list các tọa độ (x, y) tiếp theo dựa trên (Đi thẳng, Rẽ trái, Phải) đã được DQN xếp hạng.
    """
    tail = body[-1]
    safe_actions = []
    fallback_actions = [] # Lưu lại dạng tuple (hành_động, không_gian_trống)

    for next_pos in sorted_actions:
        # 1. Loại bỏ ngay hành động đâm tường
        if not (0 <= next_pos[0] < width and 0 <= next_pos[1] < height):
            continue
            
        # 2. Loại bỏ hành động đâm thân (LƯU Ý: Không tính cái đuôi vì bước tới đuôi sẽ nhích đi)
        # body[:-1] tức là lấy từ đầu đến sát cái đuôi
        if next_pos in body[:-1]:
            continue

        # Giả lập vật cản cho bước tiếp theo (đuôi đã di chuyển nên không còn là vật cản)
        future_obstacles = set(body[:-1]) 

        # 3. Ưu tiên 1: Chạy BFS tìm đuôi
        if bfs_find_tail(next_pos, tail, future_obstacles, width, height):
            safe_actions.append(next_pos)
        else:
            # 4. Ưu tiên 2: Chạy Flood Fill đếm ô trống
            space = flood_fill_space(next_pos, future_obstacles, width, height)
            fallback_actions.append((next_pos, space))

    # --- DECISION MAKING ---
    
    # Kịch bản A: Có đường an toàn về đuôi. Chọn hành động có Q-value (value) cao nhất
    if safe_actions:
        return safe_actions[0] 
        
    # Kịch bản B: Chọn hành động dẫn đến khoảng trống lớn nhất để sống sót lâu hơn.
    if fallback_actions:
        fallback_actions.sort(key=lambda x: x[1], reverse=True)
        best_fallback_action, max_space = fallback_actions[0]
        
        return best_fallback_action 
        
    return sorted_actions[0] if sorted_actions else head