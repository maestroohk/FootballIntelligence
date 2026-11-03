namespace FootballIntelligence.Web.Data;

public class MatchRow
{
    public long MatchId { get; set; }
    public string Competition { get; set; } = "";
    public string CompetitionName { get; set; } = "";
    public long Season { get; set; }
    public string MatchDate { get; set; } = "";
    public string? Kickoff { get; set; }
    public string Status { get; set; } = "";
    public long HomeTeamId { get; set; }
    public long AwayTeamId { get; set; }
    public string HomeName { get; set; } = "";
    public string AwayName { get; set; } = "";
    public long? Fthg { get; set; }
    public long? Ftag { get; set; }
    public long? Hthg { get; set; }
    public long? Htag { get; set; }
    public long? Hs { get; set; }
    public long? Aws { get; set; }
    public long? Hst { get; set; }
    public long? Ast { get; set; }
    public long? Hc { get; set; }
    public long? Ac { get; set; }
    public long? Hf { get; set; }
    public long? Af { get; set; }
    public long? Hy { get; set; }
    public long? Ay { get; set; }
    public long? Hr { get; set; }
    public long? Ar { get; set; }
    public string? Referee { get; set; }
    public double? PHome { get; set; }
    public double? PDraw { get; set; }
    public double? PAway { get; set; }
    public double? LambdaHome { get; set; }
    public double? LambdaAway { get; set; }
    public string? ModelVersion { get; set; }
    public string? Purpose { get; set; }
    public string? PredictionCreated { get; set; }
    public string? DataCutoff { get; set; }
    public string? Markets { get; set; }
    public string? PredictionFlag { get; set; }
    public string Score => Fthg is null ? "vs" : $"{Fthg} - {Ftag}";
    public string? Result => Fthg is null ? null : Fthg > Ftag ? "H" : Fthg == Ftag ? "D" : "A";
}

public class OddsRow
{
    public string Bookmaker { get; set; } = "";
    public string Timing { get; set; } = "";
    public double? Home { get; set; }
    public double? Draw { get; set; }
    public double? Away { get; set; }
    public double? Over25 { get; set; }
    public double? Under25 { get; set; }
}

public class FormRow
{
    public long MatchId { get; set; }
    public string MatchDate { get; set; } = "";
    public string Competition { get; set; } = "";
    public string Opponent { get; set; } = "";
    public string Venue { get; set; } = "";
    public long GoalsFor { get; set; }
    public long GoalsAgainst { get; set; }
    public long? ShotsFor { get; set; }
    public long? ShotsAgainst { get; set; }
    public string Outcome => GoalsFor > GoalsAgainst ? "W" : GoalsFor == GoalsAgainst ? "D" : "L";
}

public class JobRow
{
    public long JobId { get; set; }
    public string Kind { get; set; } = "";
    public string Args { get; set; } = "";
    public string Status { get; set; } = "";
    public string? Stage { get; set; }
    public long DoneUnits { get; set; }
    public long? TotalUnits { get; set; }
    public string CreatedAt { get; set; } = "";
    public string? StartedAt { get; set; }
    public string? HeartbeatAt { get; set; }
    public string? FinishedAt { get; set; }
    public string? Error { get; set; }
    public long Retries { get; set; }
    public long CancelRequested { get; set; }
    public string? Result { get; set; }
    public double? Progress => TotalUnits is > 0 ? Math.Min(1.0, (double)DoneUnits / TotalUnits.Value) : null;
    public TimeSpan? HeartbeatAge => Services.Fmt.Since(HeartbeatAt ?? StartedAt);

    public string State(bool workerAlive) => Status switch
    {
        "queued" when !workerAlive => "waiting",
        "running" when HeartbeatAge is null || HeartbeatAge > TimeSpan.FromSeconds(90) => "stalled",
        _ => Status
    };

    public string StateNote(bool workerAlive) => State(workerAlive) switch
    {
        "waiting" => "Queued, but the engine worker is offline. It starts when the worker is running.",
        "stalled" => workerAlive ? $"No heartbeat for {Services.Fmt.Duration(HeartbeatAge)}; the worker re-queues it automatically." : $"No heartbeat for {Services.Fmt.Duration(HeartbeatAge)} and the worker is offline; the process has stopped. Retry it or start the worker.",
        "running" when !workerAlive => "Running outside the worker (command line); heartbeat is current.",
        "running" => $"Heartbeat {Services.Fmt.Duration(HeartbeatAge)} ago.",
        _ => ""
    };
}

public class JobLogRow
{
    public string Ts { get; set; } = "";
    public string Level { get; set; } = "";
    public string Message { get; set; } = "";
}

public class NotificationRow
{
    public long NotificationId { get; set; }
    public string Ts { get; set; } = "";
    public string Level { get; set; } = "";
    public string Title { get; set; } = "";
    public string? Body { get; set; }
    public long Seen { get; set; }
}

public class ProviderRow
{
    public string Provider { get; set; } = "";
    public string? LastAttempt { get; set; }
    public string? LastSuccess { get; set; }
    public string? LastError { get; set; }
    public long? RequestsToday { get; set; }
    public string? RateLimit { get; set; }
    public string? Status { get; set; }
}

public class SourceRow
{
    public string Source { get; set; } = "";
    public string Url { get; set; } = "";
    public string Licence { get; set; } = "";
    public string Coverage { get; set; } = "";
    public string Notes { get; set; } = "";
}

public class CoverageRow
{
    public string Competition { get; set; } = "";
    public string Name { get; set; } = "";
    public long Seasons { get; set; }
    public long FirstSeason { get; set; }
    public long LastSeason { get; set; }
    public long Matches { get; set; }
    public long Finished { get; set; }
    public long Scheduled { get; set; }
    public long MissingHt { get; set; }
    public long MissingShots { get; set; }
    public long WithClosingOdds { get; set; }
    public long WithEvents { get; set; }
    public string? LatestResult { get; set; }
}

public class FieldRow
{
    public string Field { get; set; } = "";
    public string Status { get; set; } = "";
    public string Source { get; set; } = "";
    public string Description { get; set; } = "";
}

public class BacktestRow
{
    public long RunId { get; set; }
    public string ModelVersion { get; set; } = "";
    public string Split { get; set; } = "";
    public string CreatedAt { get; set; } = "";
    public string? TrainWindow { get; set; }
    public string? TestPeriod { get; set; }
    public long NMatches { get; set; }
    public string Metrics { get; set; } = "";
    public string? ByLeague { get; set; }
    public string? BySeason { get; set; }
    public string? Reliability { get; set; }
    public string? Notes { get; set; }
}

public class ExperimentRow
{
    public long ExperimentId { get; set; }
    public string CreatedAt { get; set; } = "";
    public string Name { get; set; } = "";
    public string Hypothesis { get; set; } = "";
    public string? Result { get; set; }
    public string? Decision { get; set; }
    public string? Details { get; set; }
}

public class ModelVersionRow
{
    public string Version { get; set; } = "";
    public string Family { get; set; } = "";
    public string Params { get; set; } = "";
    public string? TrainedThrough { get; set; }
    public string CreatedAt { get; set; } = "";
    public string Status { get; set; } = "";
    public string? Notes { get; set; }
}

public class LiveSessionRow
{
    public long LiveId { get; set; }
    public string Mode { get; set; } = "";
    public string Provider { get; set; } = "";
    public long? SbMatchId { get; set; }
    public long? MatchId { get; set; }
    public string? HomeName { get; set; }
    public string? AwayName { get; set; }
    public string Status { get; set; } = "";
    public double? Speed { get; set; }
    public string CreatedAt { get; set; } = "";
    public string? LastEventAt { get; set; }
    public string? LastUpdateAt { get; set; }
    public double? Minute { get; set; }
    public long? HomeGoals { get; set; }
    public long? AwayGoals { get; set; }
    public string? Prematch { get; set; }
    public long? JobId { get; set; }
}

public class SnapshotRow
{
    public long SnapshotId { get; set; }
    public string Ts { get; set; } = "";
    public double Clock { get; set; }
    public string Trigger { get; set; } = "";
    public long HomeGoals { get; set; }
    public long AwayGoals { get; set; }
    public double PHome { get; set; }
    public double PDraw { get; set; }
    public double PAway { get; set; }
    public double PAnotherGoal { get; set; }
    public double ExpRemaining { get; set; }
    public string? State { get; set; }
    public string? Explanation { get; set; }
}

public class LiveEventRow
{
    public string ProviderEventId { get; set; } = "";
    public string ReceivedAt { get; set; } = "";
    public long Period { get; set; }
    public double Clock { get; set; }
    public string Kind { get; set; } = "";
    public string? Side { get; set; }
    public string? PlayerName { get; set; }
    public double? Xg { get; set; }
    public string? Payload { get; set; }
    public long Superseded { get; set; }
}

public class ReplayCandidate
{
    public long SbMatchId { get; set; }
    public long MatchId { get; set; }
    public string Competition { get; set; } = "";
    public string MatchDate { get; set; } = "";
    public string HomeName { get; set; } = "";
    public string AwayName { get; set; } = "";
    public long HomeScore { get; set; }
    public long AwayScore { get; set; }
    public long Goals { get; set; }
    public long Reds { get; set; }
}

public class PlayerSummary
{
    public long PlayerId { get; set; }
    public string Name { get; set; } = "";
    public string? Team { get; set; }
    public string? Position { get; set; }
    public long Apps { get; set; }
    public long Starts { get; set; }
    public double Minutes { get; set; }
    public long Goals { get; set; }
    public long Assists { get; set; }
    public double Xg { get; set; }
    public double Xa { get; set; }
    public long Shots { get; set; }
    public long KeyPasses { get; set; }
    public long ProgressivePasses { get; set; }
    public long ProgressiveCarries { get; set; }
    public long Pressures { get; set; }
    public long Tackles { get; set; }
    public long Interceptions { get; set; }
    public long Recoveries { get; set; }
    public long Yellow { get; set; }
    public long Red { get; set; }
    public long? GkShotsFaced { get; set; }
    public long? GkGoalsConceded { get; set; }
    public double? GkXgFaced { get; set; }
    public double Per90(double v) => Minutes > 0 ? v * 90 / Minutes : 0;
}

public class PlayerMatchRow
{
    public long SbMatchId { get; set; }
    public string MatchDate { get; set; } = "";
    public string TeamName { get; set; } = "";
    public string Opponent { get; set; } = "";
    public string? Position { get; set; }
    public long Started { get; set; }
    public double Minutes { get; set; }
    public long Goals { get; set; }
    public long Assists { get; set; }
    public double Xg { get; set; }
    public double Xa { get; set; }
    public long Shots { get; set; }
    public long KeyPasses { get; set; }
    public long ProgressivePasses { get; set; }
    public long Pressures { get; set; }
    public long DaysSincePrevious { get; set; }
}

public class LineupRow
{
    public long PlayerId { get; set; }
    public string Name { get; set; } = "";
    public string Side { get; set; } = "";
    public string? Position { get; set; }
    public long Started { get; set; }
    public double MinuteOn { get; set; }
    public double MinuteOff { get; set; }
    public double Minutes { get; set; }
    public long Goals { get; set; }
    public long Assists { get; set; }
    public double Xg { get; set; }
    public double Xa { get; set; }
    public long Shots { get; set; }
    public long KeyPasses { get; set; }
    public long ProgressivePasses { get; set; }
    public long Pressures { get; set; }
    public long Yellow { get; set; }
    public long Red { get; set; }
}

public class StyleProfile
{
    public long Matches { get; set; }
    public double? GoalsFor { get; set; }
    public double? GoalsAgainst { get; set; }
    public double? ShotsFor { get; set; }
    public double? ShotsAgainst { get; set; }
    public double? SotFor { get; set; }
    public double? SotAgainst { get; set; }
    public double? CornersFor { get; set; }
    public double? CornersAgainst { get; set; }
    public double? FoulsFor { get; set; }
    public double? Yellows { get; set; }
    public double? HomeShare { get; set; }
}

public class TeamTactics
{
    public string Team { get; set; } = "";
    public string Competition { get; set; } = "";
    public long Matches { get; set; }
    public double XgFor { get; set; }
    public double XgAgainst { get; set; }
    public double ShotsFor { get; set; }
    public double ShotsAgainst { get; set; }
    public double SetPieceXgFor { get; set; }
    public double SetPieceXgAgainst { get; set; }
    public double CounterXgFor { get; set; }
    public double CounterXgAgainst { get; set; }
    public double PenaltyXgFor { get; set; }
    public double Possession { get; set; }
    public double? Ppda { get; set; }
    public double? OppPpda { get; set; }
    public double FieldTilt { get; set; }
    public double PressuresPerMatch { get; set; }
    public double BoxTouchesPerMatch { get; set; }
    public double GoalsFor { get; set; }
    public double GoalsAgainst { get; set; }
    public double PerMatch(double v) => Matches > 0 ? v / Matches : 0;
}

public class TeamInfo
{
    public long TeamId { get; set; }
    public string Name { get; set; } = "";
    public string Country { get; set; } = "";
    public string? Competition { get; set; }
    public long? Season { get; set; }
    public string? LastPlayed { get; set; }
}

public class StandingRow
{
    public long TeamId { get; set; }
    public string Name { get; set; } = "";
    public long Played { get; set; }
    public long Won { get; set; }
    public long Drawn { get; set; }
    public long Lost { get; set; }
    public long GoalsFor { get; set; }
    public long GoalsAgainst { get; set; }
    public long Points { get; set; }
}
