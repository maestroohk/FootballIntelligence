namespace FootballIntelligence.Web.Services;

public class ViewMode
{
    public bool Analyst { get; private set; }
    public event Action? Changed;

    public void Set(bool analyst)
    {
        if (Analyst == analyst) return;
        Analyst = analyst;
        Changed?.Invoke();
    }
}
