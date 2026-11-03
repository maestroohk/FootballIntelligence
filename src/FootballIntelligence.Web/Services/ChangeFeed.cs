using FootballIntelligence.Web.Data;

namespace FootballIntelligence.Web.Services;

public record FeedState(long MaxSnapshotId, string? LatestJobChange, long MaxNotificationId, long UnseenNotifications);

public class ChangeFeed(Database db, ILogger<ChangeFeed> logger) : BackgroundService
{
    public FeedState State { get; private set; } = new(0, null, 0, 0);
    public DateTime LastPoll { get; private set; }
    public event Action<FeedState, FeedState>? Changed;

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                if (db.Exists)
                {
                    var next = new FeedState(
                        db.Single<long?>("SELECT MAX(snapshot_id) FROM live_snapshots") ?? 0,
                        db.Single<string>("SELECT MAX(COALESCE(heartbeat_at, created_at)) || '|' || COUNT(*) || '|' || SUM(done_units) || '|' || GROUP_CONCAT(status) FROM (SELECT * FROM jobs ORDER BY job_id DESC LIMIT 30)") + "|" + DateTime.UtcNow.Ticks / TimeSpan.FromSeconds(30).Ticks,
                        db.Single<long?>("SELECT MAX(notification_id) FROM notifications") ?? 0,
                        db.Single<long?>("SELECT COUNT(*) FROM notifications WHERE seen=0") ?? 0);
                    LastPoll = DateTime.UtcNow;
                    if (next != State)
                    {
                        var previous = State;
                        State = next;
                        Changed?.Invoke(previous, next);
                    }
                }
            }
            catch (Exception ex)
            {
                logger.LogWarning(ex, "change feed poll failed");
            }
            await Task.Delay(TimeSpan.FromMilliseconds(1000), stoppingToken);
        }
    }
}
