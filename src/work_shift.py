import simpy

class WorkShiftManager:
    """
    Controls whether manual workers or AMRs are available based on work-shift rules.
    Provides wait-until-shift-start and active checks.
    """

    MIN_PER_DAY = 24 * 60

    def __init__(self, env: simpy.Environment, shift_cfg: dict):
        self.env = env
        self.cfg = shift_cfg

        # Basic config
        self.start = self._hhmm_to_min(shift_cfg.get("start_hhmm", "09:00"))
        self.end   = self._hhmm_to_min(shift_cfg.get("end_hhmm", "18:00"))

        self.cycle_days = int(shift_cfg.get("work_cycle_days", 7))
        self.pattern = self._build_pattern(shift_cfg)

    def _hhmm_to_min(self, s):
        hh, mm = s.split(":")
        return int(hh)*60 + int(mm)

    def _build_pattern(self, cfg):
        """Build weekly (or cycle) working / off-day pattern."""
        workdays = int(cfg.get("workdays_per_cycle", cfg.get("workdays_per_week", 7)))
        pattern = [True]*workdays + [False]*(self.cycle_days - workdays)
        return pattern

    def _state(self, now):
        day_min = now % self.MIN_PER_DAY
        day_idx = (now // self.MIN_PER_DAY) % self.cycle_days
        return day_min, day_idx

    def active(self, now):
        """Return True if shift is ON."""
        day_min, day_idx = self._state(now)
        # Off-day
        if not self.pattern[day_idx]:
            return False
        # Working hour
        if self.start <= self.end:
            return self.start <= day_min < self.end
        else:
            # overnight shift
            return (day_min >= self.start) or (day_min < self.end)

    def time_to_next_start(self, now):
        """Wait until the next shift begins."""
        day_min, day_idx = self._state(now)

        # If same day, shift starts later
        if day_min < self.start and self.pattern[day_idx]:
            return self.start - day_min

        # Otherwise search next working day
        remaining = self.MIN_PER_DAY - day_min
        for d in range(1, self.cycle_days + 1):
            idx = (day_idx + d) % self.cycle_days
            if self.pattern[idx]:
                return remaining + self.start
            remaining += self.MIN_PER_DAY

        return remaining

    def wait_if_inactive(self):
        """SimPy generator to block until shift becomes active."""
        while not self.active(int(self.env.now)):
            wait = self.time_to_next_start(int(self.env.now))
            yield self.env.timeout(wait)

    def remaining_work_minutes(self, now: int) -> int:
        if not self.active(now):
            return 0

        day_min = now % self.MIN_PER_DAY

        if self.start <= self.end:  # 09:00~18:00
            return max(0, self.end - day_min)
        else:  # 야간근무
            if day_min >= self.start:
                return (self.MIN_PER_DAY - day_min) + self.end
            else:
                return self.end - day_min
