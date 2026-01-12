//! CF (Climate and Forecast) time conventions decoder
//!
//! Parses time units like "hours since 1900-01-01 00:00:00" and converts
//! raw numeric values to microseconds since Unix epoch (1970-01-01).
//!
//! # CF Time Conventions
//!
//! The CF conventions encode time as numeric offsets from a reference date.
//! The `units` attribute specifies the time unit and reference:
//! - "seconds since 1970-01-01 00:00:00"
//! - "hours since 1900-01-01"
//! - "days since 2000-01-01 00:00:00"
//!
//! The optional `calendar` attribute specifies the calendar system:
//! - "proleptic_gregorian" (default) - standard Gregorian calendar extended backward
//! - "standard" / "gregorian" - same as proleptic_gregorian for most purposes
//! - "noleap" / "365_day" - no leap years (not yet supported)
//! - "360_day" - 12 months of 30 days each (not yet supported)

use chrono::{NaiveDate, NaiveDateTime, NaiveTime};

/// CF time attributes from Zarr metadata
#[derive(Debug, Clone)]
pub struct CFTimeAttrs {
    /// The units string, e.g., "hours since 1900-01-01 00:00:00"
    pub units: String,
    /// The calendar type, e.g., "proleptic_gregorian"
    pub calendar: Option<String>,
}

/// Parsed CF time unit components
#[derive(Debug, Clone)]
pub struct CFTimeUnit {
    /// Microseconds per time unit (e.g., 3_600_000_000 for hours)
    pub multiplier_us: i64,
    /// Reference date as microseconds since Unix epoch (1970-01-01 00:00:00 UTC)
    pub epoch_offset_us: i64,
}

impl CFTimeAttrs {
    /// Create new CF time attributes
    pub fn new(units: String, calendar: Option<String>) -> Self {
        Self { units, calendar }
    }

    /// Check if this looks like CF time encoding
    ///
    /// CF time units contain " since " to separate the unit from the reference date.
    pub fn is_time_coordinate(&self) -> bool {
        self.units.contains(" since ")
    }

    /// Parse the units string into components
    ///
    /// # Example
    /// ```
    /// use zarr_datafusion::reader::cf_time::CFTimeAttrs;
    ///
    /// let attrs = CFTimeAttrs::new("hours since 1900-01-01 00:00:00".into(), None);
    /// let unit = attrs.parse().unwrap();
    /// assert_eq!(unit.multiplier_us, 3_600_000_000);
    /// ```
    pub fn parse(&self) -> Result<CFTimeUnit, String> {
        // Split on " since "
        let parts: Vec<&str> = self.units.splitn(2, " since ").collect();
        if parts.len() != 2 {
            return Err(format!(
                "Invalid CF time units format: '{}'. Expected '<unit> since <date>'",
                self.units
            ));
        }

        let time_unit = parts[0].trim().to_lowercase();
        let reference_date = parts[1].trim();

        // Parse time unit to microseconds multiplier
        let multiplier_us = match time_unit.as_str() {
            "second" | "seconds" | "s" => 1_000_000i64,
            "minute" | "minutes" | "min" => 60_000_000i64,
            "hour" | "hours" | "h" | "hr" => 3_600_000_000i64,
            "day" | "days" | "d" => 86_400_000_000i64,
            _ => {
                return Err(format!(
                    "Unsupported time unit: '{}'. Supported: seconds, minutes, hours, days",
                    time_unit
                ))
            }
        };

        // Parse reference date to Unix epoch offset
        let epoch_offset_us = parse_reference_date(reference_date)?;

        Ok(CFTimeUnit {
            multiplier_us,
            epoch_offset_us,
        })
    }
}

/// Parse a reference date string to microseconds since Unix epoch
///
/// Supports formats:
/// - "YYYY-MM-DD"
/// - "YYYY-MM-DD HH:MM:SS"
/// - "YYYY-MM-DD HH:MM:SS.ffffff"
/// - "YYYY-MM-DDTHH:MM:SS" (ISO 8601)
fn parse_reference_date(date_str: &str) -> Result<i64, String> {
    let date_str = date_str.trim();

    // Try different formats
    let datetime = if let Ok(dt) = NaiveDateTime::parse_from_str(date_str, "%Y-%m-%d %H:%M:%S") {
        dt
    } else if let Ok(dt) = NaiveDateTime::parse_from_str(date_str, "%Y-%m-%d %H:%M:%S%.f") {
        dt
    } else if let Ok(dt) = NaiveDateTime::parse_from_str(date_str, "%Y-%m-%dT%H:%M:%S") {
        dt
    } else if let Ok(dt) = NaiveDateTime::parse_from_str(date_str, "%Y-%m-%dT%H:%M:%S%.f") {
        dt
    } else if let Ok(date) = NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
        // Date only - assume midnight
        date.and_time(NaiveTime::from_hms_opt(0, 0, 0).unwrap())
    } else {
        return Err(format!(
            "Failed to parse reference date: '{}'. Expected format like 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'",
            date_str
        ));
    };

    // Convert to microseconds since Unix epoch
    // Unix epoch is 1970-01-01 00:00:00 UTC
    Ok(datetime.and_utc().timestamp_micros())
}

/// Convert raw time values to microseconds since Unix epoch
///
/// # Arguments
/// * `values` - Raw time values from Zarr (e.g., hours since reference date)
/// * `unit` - Parsed CF time unit with multiplier and offset
///
/// # Returns
/// Vector of microseconds since Unix epoch (suitable for Arrow Timestamp)
pub fn decode_cf_time(values: &[i64], unit: &CFTimeUnit) -> Vec<i64> {
    values
        .iter()
        .map(|v| unit.epoch_offset_us + v * unit.multiplier_us)
        .collect()
}

/// Convert raw float time values to microseconds since Unix epoch
///
/// Handles fractional time values (e.g., 1.5 hours)
pub fn decode_cf_time_f64(values: &[f64], unit: &CFTimeUnit) -> Vec<i64> {
    values
        .iter()
        .map(|v| unit.epoch_offset_us + (*v * unit.multiplier_us as f64) as i64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_time_coordinate() {
        let cf = CFTimeAttrs::new("hours since 1900-01-01".into(), None);
        assert!(cf.is_time_coordinate());

        let not_cf = CFTimeAttrs::new("degrees_east".into(), None);
        assert!(!not_cf.is_time_coordinate());

        let not_cf2 = CFTimeAttrs::new("K".into(), None);
        assert!(!not_cf2.is_time_coordinate());
    }

    #[test]
    fn test_parse_hours_since_1900() {
        let cf = CFTimeAttrs::new("hours since 1900-01-01 00:00:00".into(), None);
        let unit = cf.parse().unwrap();

        // Hours to microseconds
        assert_eq!(unit.multiplier_us, 3_600_000_000);

        // 1900-01-01 is before Unix epoch (1970-01-01)
        // Difference is 70 years = 25567 days (accounting for leap years)
        // 25567 * 24 * 3600 * 1_000_000 = 2,208,988,800,000,000 microseconds
        assert!(unit.epoch_offset_us < 0);

        // Verify by checking that 0 hours since 1900 gives us 1900-01-01
        let decoded = decode_cf_time(&[0], &unit);
        let expected_1900 = NaiveDate::from_ymd_opt(1900, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_micros();
        assert_eq!(decoded[0], expected_1900);
    }

    #[test]
    fn test_parse_days_since_1970() {
        let cf = CFTimeAttrs::new("days since 1970-01-01".into(), None);
        let unit = cf.parse().unwrap();

        // Days to microseconds
        assert_eq!(unit.multiplier_us, 86_400_000_000);

        // 1970-01-01 is Unix epoch, so offset should be 0
        assert_eq!(unit.epoch_offset_us, 0);

        // Day 1 should be 1970-01-02
        let decoded = decode_cf_time(&[1], &unit);
        assert_eq!(decoded[0], 86_400_000_000); // 1 day in microseconds
    }

    #[test]
    fn test_parse_seconds_since_2000() {
        let cf = CFTimeAttrs::new("seconds since 2000-01-01".into(), None);
        let unit = cf.parse().unwrap();

        assert_eq!(unit.multiplier_us, 1_000_000);

        // 2000-01-01 offset
        let expected_2000 = NaiveDate::from_ymd_opt(2000, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_micros();
        assert_eq!(unit.epoch_offset_us, expected_2000);
    }

    #[test]
    fn test_decode_cf_time() {
        let cf = CFTimeAttrs::new("hours since 1970-01-01 00:00:00".into(), None);
        let unit = cf.parse().unwrap();

        // Decode hours 0, 1, 2
        let decoded = decode_cf_time(&[0, 1, 2], &unit);
        assert_eq!(decoded[0], 0);
        assert_eq!(decoded[1], 3_600_000_000); // 1 hour in microseconds
        assert_eq!(decoded[2], 7_200_000_000); // 2 hours in microseconds
    }

    #[test]
    fn test_decode_cf_time_f64() {
        let cf = CFTimeAttrs::new("hours since 1970-01-01".into(), None);
        let unit = cf.parse().unwrap();

        // Decode fractional hours
        let decoded = decode_cf_time_f64(&[0.0, 0.5, 1.5], &unit);
        assert_eq!(decoded[0], 0);
        assert_eq!(decoded[1], 1_800_000_000); // 0.5 hours = 30 minutes
        assert_eq!(decoded[2], 5_400_000_000); // 1.5 hours = 90 minutes
    }

    #[test]
    fn test_parse_invalid_format() {
        let cf = CFTimeAttrs::new("invalid format".into(), None);
        assert!(cf.parse().is_err());

        let cf2 = CFTimeAttrs::new("parsecs since 1900-01-01".into(), None);
        assert!(cf2.parse().is_err());
    }

    #[test]
    fn test_parse_date_only() {
        let cf = CFTimeAttrs::new("days since 2020-06-15".into(), None);
        let unit = cf.parse().unwrap();

        let expected = NaiveDate::from_ymd_opt(2020, 6, 15)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_micros();
        assert_eq!(unit.epoch_offset_us, expected);
    }

    #[test]
    fn test_era5_time_format() {
        // ERA5 uses "hours since 1900-01-01 00:00:00" with proleptic_gregorian calendar
        let cf = CFTimeAttrs::new(
            "hours since 1900-01-01 00:00:00".into(),
            Some("proleptic_gregorian".into()),
        );
        let unit = cf.parse().unwrap();

        // Value 0 should decode to 1900-01-01
        let decoded = decode_cf_time(&[0], &unit);
        let expected_1900 = NaiveDate::from_ymd_opt(1900, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_micros();
        assert_eq!(decoded[0], expected_1900);

        // Value 1 should decode to 1900-01-01 01:00:00
        let decoded = decode_cf_time(&[1], &unit);
        assert_eq!(decoded[0], expected_1900 + 3_600_000_000);
    }

    #[test]
    fn test_parse_minutes_unit() {
        let cf = CFTimeAttrs::new("minutes since 1970-01-01".into(), None);
        let unit = cf.parse().unwrap();

        assert_eq!(unit.multiplier_us, 60_000_000);
        assert_eq!(unit.epoch_offset_us, 0);

        // 60 minutes = 1 hour
        let decoded = decode_cf_time(&[60], &unit);
        assert_eq!(decoded[0], 3_600_000_000);
    }

    #[test]
    fn test_parse_unit_abbreviations() {
        // Test various unit abbreviations
        let units = vec![
            ("s", 1_000_000i64),
            ("second", 1_000_000),
            ("seconds", 1_000_000),
            ("min", 60_000_000),
            ("minute", 60_000_000),
            ("minutes", 60_000_000),
            ("h", 3_600_000_000),
            ("hr", 3_600_000_000),
            ("hour", 3_600_000_000),
            ("hours", 3_600_000_000),
            ("d", 86_400_000_000),
            ("day", 86_400_000_000),
            ("days", 86_400_000_000),
        ];

        for (unit_str, expected_multiplier) in units {
            let cf = CFTimeAttrs::new(format!("{} since 1970-01-01", unit_str), None);
            let unit = cf.parse().unwrap();
            assert_eq!(
                unit.multiplier_us, expected_multiplier,
                "Failed for unit: {}",
                unit_str
            );
        }
    }

    #[test]
    fn test_parse_iso8601_format() {
        // ISO 8601 with 'T' separator
        let cf = CFTimeAttrs::new("hours since 2020-01-15T12:30:00".into(), None);
        let unit = cf.parse().unwrap();

        let expected = NaiveDate::from_ymd_opt(2020, 1, 15)
            .unwrap()
            .and_hms_opt(12, 30, 0)
            .unwrap()
            .and_utc()
            .timestamp_micros();
        assert_eq!(unit.epoch_offset_us, expected);
    }

    #[test]
    fn test_parse_reference_with_time() {
        // Reference date with non-midnight time
        let cf = CFTimeAttrs::new("hours since 2000-06-01 06:00:00".into(), None);
        let unit = cf.parse().unwrap();

        let expected = NaiveDate::from_ymd_opt(2000, 6, 1)
            .unwrap()
            .and_hms_opt(6, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_micros();
        assert_eq!(unit.epoch_offset_us, expected);
    }

    #[test]
    fn test_decode_negative_values() {
        // Negative time values (before reference date)
        let cf = CFTimeAttrs::new("hours since 1970-01-01 00:00:00".into(), None);
        let unit = cf.parse().unwrap();

        // -1 hour should be 1969-12-31 23:00:00
        let decoded = decode_cf_time(&[-1, -24], &unit);
        assert_eq!(decoded[0], -3_600_000_000); // -1 hour
        assert_eq!(decoded[1], -86_400_000_000); // -24 hours = -1 day
    }

    #[test]
    fn test_decode_empty_array() {
        let cf = CFTimeAttrs::new("hours since 1970-01-01".into(), None);
        let unit = cf.parse().unwrap();

        let decoded = decode_cf_time(&[], &unit);
        assert!(decoded.is_empty());

        let decoded_f64 = decode_cf_time_f64(&[], &unit);
        assert!(decoded_f64.is_empty());
    }

    #[test]
    fn test_decode_large_values() {
        // Test with large values (many years)
        let cf = CFTimeAttrs::new("hours since 1900-01-01".into(), None);
        let unit = cf.parse().unwrap();

        // 1,000,000 hours ~ 114 years (should be around year 2014)
        let decoded = decode_cf_time(&[1_000_000], &unit);

        // Verify it's a reasonable timestamp (between 2000 and 2020)
        let year_2000 = NaiveDate::from_ymd_opt(2000, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_micros();
        let year_2020 = NaiveDate::from_ymd_opt(2020, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_micros();

        assert!(decoded[0] > year_2000, "Should be after year 2000");
        assert!(decoded[0] < year_2020, "Should be before year 2020");
    }

    #[test]
    fn test_case_insensitive_units() {
        // Units should be case-insensitive
        let cf_upper = CFTimeAttrs::new("HOURS since 1970-01-01".into(), None);
        let cf_mixed = CFTimeAttrs::new("Hours since 1970-01-01".into(), None);

        assert!(cf_upper.parse().is_ok());
        assert!(cf_mixed.parse().is_ok());

        let unit_upper = cf_upper.parse().unwrap();
        let unit_mixed = cf_mixed.parse().unwrap();
        assert_eq!(unit_upper.multiplier_us, unit_mixed.multiplier_us);
    }

    #[test]
    fn test_whitespace_handling() {
        // Extra whitespace should be handled
        let cf = CFTimeAttrs::new("hours   since   1970-01-01".into(), None);
        // This should still work because we split on " since "
        assert!(cf.is_time_coordinate());
    }

    #[test]
    fn test_calendar_attribute_preserved() {
        let cf = CFTimeAttrs::new("days since 2000-01-01".into(), Some("noleap".into()));
        assert_eq!(cf.calendar, Some("noleap".to_string()));
        // Note: Calendar is preserved but not currently used in calculations
        // (would need special handling for non-standard calendars)
    }
}
