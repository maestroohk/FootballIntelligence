using System.Text.Json;

namespace FootballIntelligence.Web.Data;

public class OperationsQueries(Database db)
{
    public static readonly string[] JobKinds = ["daily_cycle", "check_assets", "ingest_fixtures", "ingest_history", "ingest_statsbomb", "research_full", "predict_upcoming", "train_inplay", "evaluate_forward", "final_test", "player_study", "experiments", "rollback", "live_replay", "poll_live"];

    public List<JobRow> Jobs(int limit = 100) => db.Query<JobRow>("SELECT * FROM jobs ORDER BY job_id DESC LIMIT @limit", new { limit }).ToList();

    public List<JobRow> ActiveJobs() => db.Query<JobRow>("SELECT * FROM jobs WHERE status IN ('queued','running') ORDER BY job_id").ToList();

    public JobRow? Job(long id) => db.Single<JobRow>("SELECT * FROM jobs WHERE job_id=@id", new { id });

    public List<JobLogRow> JobLogs(long id, int limit = 300) => db.Query<JobLogRow>("SELECT ts, level, message FROM job_logs WHERE job_id=@id ORDER BY log_id DESC LIMIT @limit", new { id, limit }).ToList();

    public long Enqueue(string kind, object? args = null)
    {
        var json = args is JsonElement e ? e.GetRawText() : JsonSerializer.Serialize(args ?? new { });
        using var con = db.Open();
        using var cmd = con.CreateCommand();
        cmd.CommandText = """
            INSERT INTO jobs(kind, args, status, created_at, max_retries)
            SELECT $kind, $args, 'queued', $now, 2
            WHERE NOT EXISTS (SELECT 1 FROM jobs WHERE kind=$kind AND args=$args AND status IN ('queued','running') AND cancel_requested=0);
            SELECT COALESCE((SELECT job_id FROM jobs WHERE kind=$kind AND args=$args AND status IN ('queued','running') ORDER BY job_id LIMIT 1), last_insert_rowid());
            """;
        cmd.Parameters.AddWithValue("$kind", kind);
        cmd.Parameters.AddWithValue("$args", json);
        cmd.Parameters.AddWithValue("$now", Now());
        return (long)cmd.ExecuteScalar()!;
    }

    public void Cancel(long id) => db.Execute("""
        UPDATE jobs SET cancel_requested=1,
               status=CASE WHEN status='queued' OR COALESCE(heartbeat_at, started_at) < @stale THEN 'cancelled' ELSE status END,
               finished_at=CASE WHEN status='queued' OR COALESCE(heartbeat_at, started_at) < @stale THEN @now ELSE finished_at END
        WHERE job_id=@id AND status IN ('queued','running')
        """, new { id, now = Now(), stale = DateTime.UtcNow.AddSeconds(-90).ToString("yyyy-MM-ddTHH:mm:ss") });

    public long Retry(long id)
    {
        var job = Job(id) ?? throw new InvalidOperationException("job not found");
        return Enqueue(job.Kind, JsonSerializer.Deserialize<JsonElement>(job.Args));
    }

    public List<NotificationRow> Notifications(int limit = 30) => db.Query<NotificationRow>("SELECT * FROM notifications ORDER BY notification_id DESC LIMIT @limit", new { limit }).ToList();

    public void MarkSeen(long id) => db.Execute("UPDATE notifications SET seen=1 WHERE notification_id=@id", new { id });

    public void MarkAllSeen() => db.Execute("UPDATE notifications SET seen=1 WHERE seen=0");

    public List<ProviderRow> Providers() => db.Query<ProviderRow>("SELECT * FROM provider_health ORDER BY provider").ToList();

    public List<SourceRow> Sources() => db.Query<SourceRow>("SELECT * FROM sources").ToList();

    public List<FieldRow> Fields() => db.Query<FieldRow>("SELECT * FROM field_catalog ORDER BY status, field").ToList();

    public List<CoverageRow> Coverage() => db.Query<CoverageRow>("""
        SELECT m.competition, c.name, COUNT(DISTINCT m.season) seasons, MIN(m.season) first_season, MAX(m.season) last_season,
               COUNT(*) matches, SUM(m.status='finished') finished, SUM(m.status='scheduled') scheduled,
               SUM(m.status='finished' AND m.hthg IS NULL) missing_ht, SUM(m.status='finished' AND m.hs IS NULL) missing_shots,
               (SELECT COUNT(DISTINCT o.match_id) FROM odds o JOIN matches x ON x.match_id=o.match_id WHERE x.competition=m.competition AND o.timing='close') with_closing_odds,
               (SELECT COUNT(*) FROM sb_matches s WHERE s.competition=m.competition AND s.processed=1) with_events,
               MAX(CASE WHEN m.status='finished' THEN m.match_date END) latest_result
        FROM matches m JOIN competitions c ON c.code=m.competition
        GROUP BY m.competition ORDER BY c.tier, m.competition
        """).ToList();

    public WorkerStatus WorkerStatus()
    {
        var hb = db.Kv("worker_heartbeat");
        DateTime? beat = hb is null ? null : DateTime.Parse(JsonSerializer.Deserialize<string>(hb)!, null, System.Globalization.DateTimeStyles.AdjustToUniversal);
        return new WorkerStatus(beat, beat is not null && DateTime.UtcNow - beat.Value < TimeSpan.FromSeconds(30), db.Exists);
    }

    public List<LiveSessionRow> LiveSessions(int limit = 50) => db.Query<LiveSessionRow>("SELECT * FROM live_sessions ORDER BY live_id DESC LIMIT @limit", new { limit }).ToList();

    public LiveSessionRow? LiveSession(long id) => db.Single<LiveSessionRow>("SELECT * FROM live_sessions WHERE live_id=@id", new { id });

    public List<SnapshotRow> Snapshots(long id) => db.Query<SnapshotRow>("SELECT * FROM live_snapshots WHERE live_id=@id ORDER BY snapshot_id", new { id }).ToList();

    public List<LiveEventRow> LiveEvents(long id) => db.Query<LiveEventRow>(
        "SELECT * FROM live_events WHERE live_id=@id AND kind NOT IN ('minute_stats') ORDER BY period, clock, provider_seq", new { id }).ToList();

    public List<ReplayCandidate> ReplayCandidates(string? search, int limit = 60) => db.Query<ReplayCandidate>("""
        SELECT s.sb_match_id, s.match_id, s.competition, s.match_date, s.home_name, s.away_name, s.home_score, s.away_score,
               s.home_score + s.away_score goals,
               (SELECT COUNT(*) FROM sb_events e WHERE e.sb_match_id=s.sb_match_id AND e.kind='card' AND e.detail IN ('Red Card','Second Yellow')) reds
        FROM sb_matches s
        WHERE s.processed=1 AND s.match_id IS NOT NULL
          AND (@search IS NULL OR s.home_name LIKE '%'||@search||'%' OR s.away_name LIKE '%'||@search||'%')
          AND EXISTS (SELECT 1 FROM predictions p WHERE p.match_id=s.match_id AND p.purpose='backtest')
        ORDER BY s.match_date DESC LIMIT @limit
        """, new { search, limit }).ToList();

    public bool InPlayTrained() => db.Single<long>("SELECT COUNT(*) FROM live_model WHERE status='champion'") > 0;

    public JsonElement? Assets() => db.Kv("assets_status") is { } v ? JsonDocument.Parse(v).RootElement.Clone() : null;

    public static string Now() => DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.ffffffZ");
}

public record WorkerStatus(DateTime? LastHeartbeat, bool Alive, bool DatabaseExists);
