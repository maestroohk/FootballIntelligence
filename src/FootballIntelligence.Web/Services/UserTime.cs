using System.Globalization;
using FootballIntelligence.Web.Data;

namespace FootballIntelligence.Web.Services;

public class UserTime
{
    public static readonly (string Id, string Label)[] Choices =
    [
        ("", "This computer"), ("Europe/London", "UK"), ("Europe/Paris", "Central Europe"), ("Europe/Athens", "Eastern Europe"),
        ("America/New_York", "US Eastern"), ("America/Los_Angeles", "US Pacific"), ("Africa/Johannesburg", "South Africa"),
        ("Asia/Dubai", "Gulf"), ("Asia/Kolkata", "India"), ("Australia/Sydney", "Sydney"), ("UTC", "UTC"),
    ];

    private static readonly Dictionary<string, string> LeagueZones = new()
    {
        ["ENG"] = "Europe/London", ["SCO"] = "Europe/London", ["ESP"] = "Europe/Madrid", ["GER"] = "Europe/Berlin", ["ITA"] = "Europe/Rome",
        ["FRA"] = "Europe/Paris", ["NED"] = "Europe/Amsterdam", ["POR"] = "Europe/Lisbon", ["BEL"] = "Europe/Brussels", ["GRE"] = "Europe/Athens",
        ["TUR"] = "Europe/Istanbul",
    };

    public string ZoneId { get; private set; } = "";
    public TimeZoneInfo Zone { get; private set; } = TimeZoneInfo.Local;
    public event Action? Changed;

    public string Label => ZoneId == "" ? "local" : Choices.FirstOrDefault(c => c.Id == ZoneId).Label ?? ZoneId;

    public void Set(string? id)
    {
        id ??= "";
        if (id == ZoneId) return;
        try { Zone = id == "" ? TimeZoneInfo.Local : TimeZoneInfo.FindSystemTimeZoneById(id); }
        catch (Exception ex) when (ex is TimeZoneNotFoundException or InvalidTimeZoneException) { id = ""; Zone = TimeZoneInfo.Local; }
        ZoneId = id;
        Changed?.Invoke();
    }

    public static DateTime? KickoffUtc(MatchRow m)
    {
        if (string.IsNullOrEmpty(m.Kickoff) || !DateTime.TryParse($"{m.MatchDate} {m.Kickoff}", CultureInfo.InvariantCulture, DateTimeStyles.None, out var local)) return null;
        var zoneId = m.Source == "football-data.co.uk" ? "Europe/London" : LeagueZones.GetValueOrDefault(Fmt.Country(m.Competition), "Europe/London");
        try { return TimeZoneInfo.ConvertTimeToUtc(local, TimeZoneInfo.FindSystemTimeZoneById(zoneId)); }
        catch (Exception ex) when (ex is TimeZoneNotFoundException or ArgumentException) { return null; }
    }

    public DateTime? Kickoff(MatchRow m) => KickoffUtc(m) is { } utc ? TimeZoneInfo.ConvertTimeFromUtc(utc, Zone) : null;

    public string Time(MatchRow m) => Kickoff(m)?.ToString("HH:mm", CultureInfo.InvariantCulture) ?? "TBC";

    public string Date(MatchRow m) => (Kickoff(m) ?? DateTime.Parse(m.MatchDate, CultureInfo.InvariantCulture)).ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);

    public DateTime Today => TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow, Zone).Date;
}
