"""
Shared look of the paper figures, so every figure follows the same choices
(decided on Fig 2, 12 Nov 2026; the list is also in notes/HUB.md, "Figure style").

- Text: DejaVu Sans, black, tick labels included: titles 8 pt, axis labels 7 pt, tick
  labels and legends 6.5 pt.
- Figure title (the lake or site name): bold, 9 pt, centred above the panels.
- Panel titles: "a) Title", regular weight, centred; wrapped onto two lines when
  wider than the panel. Elevation change is written "dh"; "vs", not "against".
- Panels side by side in one row where they fit; figure 7.1 in wide; PNG at 300 dpi + PDF.
- Map axes: "X (km) - EPSG:3413" below and "Y (km) - EPSG:3413" on the left; longitude
  above and latitude on the right at the SAME tick positions (three per side, two
  decimals), read along the top and right edges.
- Colours: SD of elevation = magma from 0; elevation change = RdBu, symmetric about 0
  (red = lowering); no data = light grey.
- Colour bars: horizontal, under the map, thin and 70 % of the map's width, whole-number
  ticks, label close to the bar.
- Tick labels sit close to their ticks.
- Outlines: manual = black, background ring = blue dashed, SD outline = green, each with
  a thin white casing so it reads on dark and light maps. Several SD outlines at once:
  one green ramp, dark for the lowest threshold (outermost) to light for the highest.
- Legend: inside a panel corner on a white box with no border, 5.5 pt; not above the figure.
"""
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

INK = "#0b0b0b"
C_MANUAL = "#0b0b0b"
C_RING = "#2a78d6"
C_SD = "#1baf7a"
C_OTHER = "#6f6e69"
C_NODATA = "#e9e8e4"
C_GRID = "#dddcd6"
RING_DASH = (0, (4, 2))

CMAP_SD = "magma"
CMAP_DZ = "RdBu"

FIG_WIDTH_IN = 7.1
FONT_PT = 8        # titles
LABEL_PT = 7       # axis and colour-bar labels
SMALL_PT = 6.5     # tick labels, legends
TICK_PAD = 1.5     # points between a tick and its label
CBAR_HEIGHT_RATIO = 0.036  # colour-bar row height relative to the map row
SD_RAMP = ("#0b5d3f", "#15885f", "#1baf7a", "#6fd0a8", "#b5ead3")  # dark = lowest threshold
SAVE_DPI = 300


def apply():
    """Sets the shared matplotlib defaults. Call once before building a figure."""
    plt.rcParams.update({
        "font.size": FONT_PT, "font.family": "DejaVu Sans", "axes.linewidth": 0.6,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.labelsize": LABEL_PT, "xtick.labelsize": SMALL_PT, "ytick.labelsize": SMALL_PT,
        "legend.fontsize": SMALL_PT, "xtick.major.pad": TICK_PAD, "ytick.major.pad": TICK_PAD,
    })


def figure_title(fig, text):
    """Bold title above all panels (the lake or site name)."""
    return fig.suptitle(text, fontsize=FONT_PT + 1, fontweight="bold")


def panel_title(ax, letter, text, pad=6, max_overhang=1.0):
    """
    "a) Title", regular weight, centred. A title wider than max_overhang x its panel is
    wrapped onto two lines, before an opening bracket if it has one, otherwise at the
    middle space. Raise max_overhang when there is free room beside the panel's title.
    """
    title = ax.set_title(f"{letter}) {text}", loc="center", fontsize=FONT_PT, fontweight="normal", pad=pad)
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if title.get_window_extent(renderer).width > max_overhang * ax.get_window_extent(renderer).width:
        full = f"{letter}) {text}"
        if " (" in full:
            wrapped = full.replace(" (", "\n(", 1)
        else:
            spaces = [i for i, ch in enumerate(full) if ch == " "]
            cut = min(spaces, key=lambda i: abs(i - len(full) / 2)) if spaces else None
            wrapped = full if cut is None else full[:cut] + "\n" + full[cut + 1:]
        title.set_text(wrapped)
    return title


def geo_axes(ax, extent, label_left=True, label_right=True, n_ticks=3):
    """
    Map coordinates on all four sides, with the same tick positions in both systems:
    EPSG:3413 in km below and on the left, longitude above and latitude on the right
    (two decimals). Longitude is read along the top edge and latitude along the right
    edge, so the values are slightly approximate elsewhere in the panel.

    extent: (left, right, bottom, top) in EPSG:3413 metres.
    label_left / label_right: set False on inner sides of maps that share an extent.
    """
    from pyproj import Transformer
    to_geo = Transformer.from_crs("EPSG:3413", "EPSG:4326", always_xy=True)
    x0, x1, y0, y1 = extent

    def ticks(lo, hi):
        # the coarsest round step (km) that puts at least n_ticks ticks inside the panel; of
        # those, the n_ticks nearest the middle. Coarse steps keep the labels from touching
        for step in (500, 250, 200, 150, 100, 50, 25, 20, 15, 10, 5, 2.5, 2, 1.5, 1, 0.5, 0.25, 0.2,
                     0.15, 0.1, 0.05):
            vals = np.arange(np.ceil(lo / 1000 / step), np.floor(hi / 1000 / step) + 1) * step * 1000
            if len(vals) >= n_ticks:
                windows = [vals[i:i + n_ticks] for i in range(len(vals) - n_ticks + 1)]
                return min(windows, key=lambda w: abs(w.mean() - (lo + hi) / 2))
        return vals

    def km(vals):
        decimals = next((d for d in (0, 1, 2) if np.allclose(vals / 1000, np.round(vals / 1000, d))), 2)
        return [f"{v / 1000:,.{decimals}f}".replace(",", " ").replace("-", "−") for v in vals]

    xt, yt = ticks(x0, x1), ticks(y0, y1)
    lon, _ = to_geo.transform(xt, np.full_like(xt, y1))
    _, lat = to_geo.transform(np.full_like(yt, x1), yt)

    ax.set_xticks(xt)
    ax.set_xticklabels(km(xt))
    ax.set_xlabel("X (km) - EPSG:3413")
    ax.set_yticks(yt)
    ax.set_yticklabels(km(yt) if label_left else [])
    if label_left:
        ax.set_ylabel("Y (km) - EPSG:3413")
    ax.tick_params(labelsize=SMALL_PT, length=3, pad=TICK_PAD)

    top = ax.secondary_xaxis("top")
    top.set_xticks(xt)
    top.set_xticklabels([f"{abs(v):.2f}°{'W' if v < 0 else 'E'}" for v in lon])
    top.tick_params(labelsize=SMALL_PT, length=3, pad=TICK_PAD)
    side = ax.secondary_yaxis("right")
    side.set_yticks(yt)
    side.set_yticklabels([f"{abs(v):.2f}°{'N' if v >= 0 else 'S'}" for v in lat] if label_right else [])
    side.tick_params(labelsize=SMALL_PT, length=3, pad=TICK_PAD)


def map_colorbar(fig, im, cell_ax, label, extend="neither", width=0.7):
    """
    Horizontal colour bar with whole-number ticks, drawn inside cell_ax (an empty axes
    under the map, e.g. the second row of a gridspec).
    """
    cell_ax.axis("off")
    cax = cell_ax.inset_axes([(1 - width) / 2, 0.0, width, 1.0])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend=extend)
    cb.locator = MaxNLocator(nbins=5, steps=[1, 2, 3, 5, 10], integer=True)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=SMALL_PT, length=2.5, pad=TICK_PAD)
    cb.set_label(label, labelpad=1.5)
    return cb


def draw_outline(ax, geom, color, linewidth=1.3, linestyle="-", casing=True):
    """
    Polygon outline with a thin white casing (reads on dark and on light maps). Turn the
    casing off when several nested outlines are drawn together, or they merge into a band.
    """
    for poly in getattr(geom, "geoms", [geom]):
        for line in [poly.exterior, *poly.interiors]:
            xs, ys = line.xy
            if casing:
                ax.plot(xs, ys, color="white", linewidth=linewidth + 0.9, solid_capstyle="round", zorder=5)
            ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle, zorder=6)


def outline_handle(color, label, linestyle="-"):
    """Legend entry matching draw_outline."""
    return Line2D([], [], color=color, linewidth=1.3, linestyle=linestyle, label=label,
                  path_effects=[pe.Stroke(linewidth=2.7, foreground="white"), pe.Normal()])


def ramp_handle(colors, label):
    """One legend entry standing for several outlines drawn in a colour ramp."""
    return tuple(Line2D([], [], color=c, linewidth=1.6) for c in colors), label


def corner_legend(ax, handles, loc="upper left"):
    """
    Legend in a panel corner, on an opaque white box with no border.
    handles: Line2D entries (outline_handle) and/or (lines, label) pairs from ramp_handle.
    """
    from matplotlib.legend_handler import HandlerTuple
    items = [h if isinstance(h, tuple) else (h, h.get_label()) for h in handles]
    legend = ax.legend([h for h, _ in items], [lab for _, lab in items], loc=loc,
                       fontsize=SMALL_PT - 1, frameon=True, framealpha=1.0, facecolor="white",
                       edgecolor="none", borderpad=0.4, handlelength=2.0, labelspacing=0.3,
                       borderaxespad=0.0, fancybox=False,
                       handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)})
    legend.set_zorder(10)
    return legend


def save(fig, stem):
    """PNG (300 dpi) and PDF, trimmed."""
    fig.savefig(stem + ".png", dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(stem + ".pdf", bbox_inches="tight", pad_inches=0.05)
