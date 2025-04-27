import time

class FallDetection:
    def __init__(self, fall_threshold=0.75, cooldown_time=15, max_frames=100):
        self.fall_threshold = fall_threshold  # Threshold for how many frames must be "fall" to trigger an alarm
        self.cooldown_time = cooldown_time  # Time to wait before allowing another fall detection
        self.max_frames = max_frames  # Number of frames to consider for prediction
        self.frame_buffer = []  # Buffer to store the last frames (fall/non-fall)
        self.last_fall_time = None  # Time when the last fall occurred

    def add_frame(self, is_fall):
        """Add a new frame to the buffer and clean up if necessary."""
        if len(self.frame_buffer) >= self.max_frames:
            self.frame_buffer.pop(0)  # Remove the oldest frame to make room

        self.frame_buffer.append(is_fall)

    def should_trigger_alarm(self):
        """Check if the fall alarm should be triggered based on the buffer."""
        if len(self.frame_buffer) < self.max_frames:
            return False  # Not enough frames to decide

        # Count how many frames are falls
        fall_count = sum(self.frame_buffer)
        fall_ratio = fall_count / self.max_frames

        if fall_ratio >= self.fall_threshold:
            return True  # Alarm should be triggered

        return False

    def can_detect_fall(self):
        """Check if the system is in cooldown (i.e., wait before detecting another fall)."""
        if self.last_fall_time is None:
            return True  # No fall detected before, so can detect a fall

        # If the cooldown period has passed, we can detect the next fall
        if time.time() - self.last_fall_time >= self.cooldown_time:
            return True

        return False

    def record_fall(self):
        """Record the time of the fall detection."""
        self.last_fall_time = time.time()

    def reset(self):
        """Reset the state of the fall detection."""
        self.frame_buffer.clear()
        self.last_fall_time = None
