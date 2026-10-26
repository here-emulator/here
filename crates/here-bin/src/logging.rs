use std::time;

use clap::ValueEnum;
use flexi_logger::Cleanup;
use flexi_logger::Criterion;
use flexi_logger::Duplicate;
use flexi_logger::FileSpec;
use flexi_logger::LogSpecBuilder;
use flexi_logger::Logger;
use flexi_logger::LoggerHandle;
use flexi_logger::Naming;
use flexi_logger::WriteMode;
use log::LevelFilter;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    pub fn to_level_filter(&self) -> LevelFilter {
        match self {
            LogLevel::Trace => LevelFilter::Trace,
            LogLevel::Debug => LevelFilter::Debug,
            LogLevel::Info => LevelFilter::Info,
            LogLevel::Warn => LevelFilter::Warn,
            LogLevel::Error => LevelFilter::Error,
        }
    }
}

fn duration_to_str_min(dur: time::Duration) -> String {
    format!(
        "{:02}:{:02}.{:03}",
        (dur.as_secs() % 3600) / 60,
        dur.as_secs() % 60,
        dur.subsec_millis()
    )
}

fn format_msg_elapsed_time(
    w: &mut dyn std::io::Write,
    _now: &mut flexi_logger::DeferredNow,
    record: &log::Record,
) -> Result<(), std::io::Error> {
    static START_DATE: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let start_time = START_DATE.get_or_init(|| std::time::Instant::now());
    let elapsed = start_time.elapsed();

    write!(
        w,
        "[{}][{:5}] ",
        duration_to_str_min(elapsed),
        record.level(),
        // record.module_path().unwrap_or("<unnamed>"),
        // record.file().unwrap_or("<unnamed>"),
        // record.line().unwrap_or(0),
    )?;

    write!(w, "{}", &record.args())
}

/// Initialize the logger.
/// Must keep the [`LoggerHandle`] (returned value) alive up to the very end of your program
/// to ensure that all buffered log lines are flushed out.
#[must_use]
pub fn init(level: LogLevel) -> LoggerHandle {
    let mut builder = LogSpecBuilder::new();
    builder.default(level.to_level_filter());

    Logger::with(builder.build())
        .log_to_file(
            FileSpec::default()
                .directory("logs")
                .basename("emulator")
                .suffix("log"),
        )
        .rotate(
            Criterion::Size(10_000_000), // 10 MB
            Naming::Numbers,
            Cleanup::KeepLogFiles(3),
        )
        .write_mode(WriteMode::BufferAndFlush)
        .duplicate_to_stderr(Duplicate::Error)
        .format_for_files(format_msg_elapsed_time)
        .start()
        .unwrap()
}

#[macro_export]
macro_rules! error_and_panic {
    ($($arg:tt)*) => {
        {
            log::error!($($arg)*);
            panic!();
        }
    };
}
