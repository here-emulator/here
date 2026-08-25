use std::{cell::Cell, rc::Rc};

pub trait Clock {
    fn now(&self) -> u64;
}

/// A clock derived lazily from the board's cycle counter.
#[derive(Clone)]
pub struct VirtualClock {
    cycles: Rc<Cell<u64>>,
}

impl VirtualClock {
    pub fn new(cycles: Rc<Cell<u64>>) -> Self {
        Self { cycles }
    }
}

impl Clock for VirtualClock {
    fn now(&self) -> u64 {
        self.cycles.get() >> 3
    }
}

struct ScheduledTask {
    due: u64,
    seq: u64,
    callback: Box<dyn FnMut()>,
}

impl ScheduledTask {
    fn new<F: FnMut() + 'static>(seq: u64, callback: F) -> Self {
        Self {
            due: u64::MAX,
            seq: seq,
            callback: Box::new(callback),
        }
    }
}

pub struct Timer<C: Clock> {
    seq: u64,
    tasks: Vec<ScheduledTask>,
    clock: C,
}

impl<C: Clock> Timer<C> {
    pub fn new(clock: C) -> Self {
        Self {
            seq: 0,
            tasks: Vec::new(),
            clock,
        }
    }

    /// Register a new task without setting a due, returning the sequence ID of the task.
    #[must_use]
    pub fn register<F>(&mut self, callback: F) -> u64
    where
        F: FnMut() + 'static,
    {
        let st = ScheduledTask::new(self.seq, callback);
        self.seq += 1;
        self.tasks.push(st);

        self.seq - 1
    }

    pub fn build(&mut self) {
        self.tasks.sort_unstable_by_key(|task| task.due);
    }

    /// Set the due time, use [`Timer::set_delay`] for a certain delay.
    ///
    /// NOTE: If you want to change multiply tasks, use [`Timer::guard`] instead.
    pub fn set_due(&mut self, seq: u64, new_due: u64) {
        self.guard().set_due(seq, new_due);
    }

    /// Set the due time to the current time + given delay.
    ///
    /// NOTE: If you want to change multiply tasks, use [`Timer::guard`] instead.
    pub fn set_delay(&mut self, seq: u64, delay: u64) {
        let now = self.clock.now();
        self.set_due(seq, now.saturating_add(delay));
    }

    /// Start a guard that allows batching multiple changes without rebuilding on each change.
    /// When the returned `TimerGuard` is dropped, the timer will be rebuilt.
    pub fn guard(&mut self) -> TimerGuard<'_, C> {
        TimerGuard { timer: self }
    }

    /// Run all tasks whose due time is <= the timer's clock `now()`.
    pub fn tick(&mut self) {
        let now = self.clock.now();

        self.tasks
            .iter_mut()
            .take_while(|task| task.due <= now)
            .for_each(|task| {
                (task.callback)();
                task.due = u64::MAX;
            });

        self.build();
    }

    /// Peek the next scheduled due time, if any.
    pub fn next_due(&self) -> Option<u64> {
        self.tasks.first().map(|s| s.due)
    }
}

/// RAII guard returned by [`Timer::guard()`].
pub struct TimerGuard<'a, C: Clock> {
    timer: &'a mut Timer<C>,
}

impl<'a, C: Clock> TimerGuard<'a, C> {
    /// See [Timer::register].
    pub fn register<F>(&mut self, callback: F) -> u64
    where
        F: FnMut() + 'static,
    {
        self.timer.register(callback)
    }

    /// See [`Timer::set_due`].
    pub fn set_due(&mut self, seq: u64, new_due: u64) {
        self.timer
            .tasks
            .iter_mut()
            .find(|task| task.seq == seq)
            .map(|task| task.due = new_due);
    }

    /// See [`Timer::set_delay`].
    pub fn set_delay(&mut self, seq: u64, delay: u64) {
        let now = self.timer.clock.now();
        self.set_due(seq, now.saturating_add(delay));
    }
}

impl<'a, C: Clock> Drop for TimerGuard<'a, C> {
    fn drop(&mut self) {
        self.timer.build();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedClock(u64);

    impl Clock for FixedClock {
        fn now(&self) -> u64 {
            self.0
        }
    }

    #[test]
    fn virtual_clock_is_derived_lazily_from_cycles() {
        let cycles = Rc::new(Cell::new(0));
        let clock = VirtualClock::new(cycles.clone());

        cycles.set(7);
        assert_eq!(clock.now(), 0);

        cycles.set(8);
        assert_eq!(clock.now(), 1);

        cycles.set(80);
        assert_eq!(clock.now(), 10);
    }

    #[test]
    fn timer_accepts_any_clock_implementation() {
        let called = Rc::new(Cell::new(false));
        let mut timer = Timer::new(FixedClock(10));
        let task = timer.register({
            let called = called.clone();
            move || called.set(true)
        });

        timer.set_due(task, 10);
        timer.tick();

        assert!(called.get());
    }
}
