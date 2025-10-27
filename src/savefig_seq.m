function path_out = savefig_seq(action, arg1, arg2)
% SAVEFIG_SEQ  Simple sequential figure saver with run-scoped folder.
% Usage:
%   savefig_seq('init', outdir)                     % set (and create) output dir, reset index to 1
%   path = savefig_seq('save', fig_handle, name)    % save fig as "<NN>_<name>.png" in outdir
%
% Notes:
%   - Uses a persistent counter; call 'init' ONCE per run.
%   - 'name' should be a short base filename WITHOUT extension.
%   - Uses exportgraphics (raster PNG, 300 DPI). Change as needed.

persistent OUTDIR IDX
switch lower(action)
    case 'init'
        OUTDIR = char(arg1);
        if ~exist(OUTDIR,'dir'), mkdir(OUTDIR); end
        IDX = 1;
        path_out = OUTDIR;

    case 'save'
        fig  = arg1;
        base = char(arg2);
        if isempty(OUTDIR) || isempty(IDX)
            % error('savefig_seq: Not initialized. Call savefig_seq(''init'', outdir) first.');
            return
        end
        safe = regexprep(base, '\s+', '_');
        fname = sprintf('%02d_%s.png', IDX, safe);
        fpath = fullfile(OUTDIR, fname);
        
        % Can also export pdf/vector images if you prefer
        % exportgraphics(fig, fullfile(OUTDIR, [fname(1:end-4) '.pdf']), 'ContentType','vector');
        
        exportgraphics(fig, fpath, 'Resolution', 300);
        path_out = fpath;
        IDX = IDX + 1;

    otherwise
        error('savefig_seq: unknown action "%s".', action);
end
