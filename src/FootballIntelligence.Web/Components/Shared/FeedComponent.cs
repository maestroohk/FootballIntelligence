using FootballIntelligence.Web.Services;
using Microsoft.AspNetCore.Components;

namespace FootballIntelligence.Web.Components.Shared;

public abstract class FeedComponent : ComponentBase, IDisposable
{
    [Inject] protected ChangeFeed Feed { get; set; } = default!;

    protected override void OnInitialized()
    {
        Feed.Changed += Handle;
        Load();
    }

    private void Handle(FeedState previous, FeedState next)
    {
        if (!ShouldRefresh(previous, next)) return;
        _ = InvokeAsync(() =>
        {
            try { Load(); } catch { }
            StateHasChanged();
        });
    }

    protected virtual bool ShouldRefresh(FeedState previous, FeedState next) => true;

    protected abstract void Load();

    public virtual void Dispose()
    {
        Feed.Changed -= Handle;
        GC.SuppressFinalize(this);
    }
}
