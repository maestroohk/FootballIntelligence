using System.Diagnostics;
using FootballIntelligence.Web.Data;

namespace FootballIntelligence.Web.Services;

public class EngineSupervisor(Database db, ILogger<EngineSupervisor> logger) : BackgroundService
{
    private Process? worker;
    public int Restarts { get; private set; }
    public string? LastError { get; private set; }
    public bool Managed => db.Options.StartWorker;
    public bool Running => worker is { HasExited: false };

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        if (!db.Options.StartWorker)
        {
            logger.LogInformation("Engine worker management disabled; expecting an external worker");
            return;
        }
        while (!stoppingToken.IsCancellationRequested)
        {
            if ((worker is null || worker.HasExited) && ExternalWorkerAlive())
            {
                Adopted = true;
            }
            else if (worker is null || worker.HasExited)
            {
                Adopted = false;
                if (worker is not null)
                {
                    Restarts++;
                    logger.LogWarning("Engine worker exited with code {Code}; restarting", worker.ExitCode);
                }
                Start();
            }
            await Task.Delay(TimeSpan.FromSeconds(5), stoppingToken);
        }
    }

    public bool Adopted { get; private set; }

    private bool ExternalWorkerAlive()
    {
        try
        {
            var raw = db.Kv("worker_heartbeat");
            if (raw is null) return false;
            var beat = DateTime.Parse(System.Text.Json.JsonSerializer.Deserialize<string>(raw)!, null, System.Globalization.DateTimeStyles.AdjustToUniversal);
            return DateTime.UtcNow - beat < TimeSpan.FromSeconds(10);
        }
        catch
        {
            return false;
        }
    }

    private void Start()
    {
        try
        {
            if (!File.Exists(db.PythonFile))
            {
                LastError = $"Python not found at {db.PythonFile}";
                logger.LogError("{Error}", LastError);
                return;
            }
            var info = new ProcessStartInfo(db.PythonFile, "-m fie worker")
            {
                WorkingDirectory = db.EngineDirectory,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            };
            info.Environment["FIE_DB"] = db.DatabaseFile;
            info.Environment["FIE_PROFILE"] = db.Options.Profile;
            info.Environment["PYTHONIOENCODING"] = "utf-8";
            worker = Process.Start(info);
            if (worker is null) return;
            worker.PriorityClass = ProcessPriorityClass.BelowNormal;
            worker.OutputDataReceived += (_, e) => { if (e.Data is not null) logger.LogInformation("[engine] {Line}", e.Data); };
            worker.ErrorDataReceived += (_, e) => { if (e.Data is not null) logger.LogWarning("[engine] {Line}", e.Data); };
            worker.BeginOutputReadLine();
            worker.BeginErrorReadLine();
            LastError = null;
            logger.LogInformation("Engine worker started (pid {Pid})", worker.Id);
        }
        catch (Exception ex)
        {
            LastError = ex.Message;
            logger.LogError(ex, "Failed to start engine worker");
        }
    }

    public override async Task StopAsync(CancellationToken cancellationToken)
    {
        await base.StopAsync(cancellationToken);
        try
        {
            if (worker is { HasExited: false }) worker.Kill(entireProcessTree: false);
        }
        catch (Exception ex)
        {
            logger.LogWarning(ex, "Failed to stop engine worker");
        }
    }
}
