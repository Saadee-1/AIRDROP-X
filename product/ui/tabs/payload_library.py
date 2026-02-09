"""
Payload Library tab. Label "Payload" -> click -> 3 categories -> click category -> payloads list.
High-contrast text on dark background. Text fits inside panels.
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
import matplotlib.patches as mpatches

# Dark panels, bright text so it's readable
_BG = "#0c0e0c"
_PANEL = "#141814"
_ACCENT = "#00ff41"
_LABEL = "#90e090"
_TEXT = "#e0ece0"
_BORDER = "#3a4a3a"
_INPUT_BG = "#1c201c"

CATEGORIES = ["Humanitarian", "Research", "Training / Inert"]

PAYLOAD_LIBRARY = [
    {"id": "medkit", "name": "Medical Kit", "category": "Humanitarian",
     "description": "Compact medical supply container for field use.",
     "mass": 2.0, "reference_area": 0.02, "drag_coefficient": 0.80,
     "shape": "box", "dimensions": {"length": 0.25, "width": 0.20, "height": 0.12}},
    {"id": "foodpack", "name": "Food Pack", "category": "Humanitarian",
     "description": "Durable ration pack for aerial delivery.",
     "mass": 3.0, "reference_area": 0.025, "drag_coefficient": 0.75,
     "shape": "box", "dimensions": {"length": 0.30, "width": 0.22, "height": 0.15}},
    {"id": "water", "name": "Water Canister", "category": "Humanitarian",
     "description": "Sealed water container, cylindrical.",
     "mass": 5.0, "reference_area": 0.03, "drag_coefficient": 0.70,
     "shape": "cylinder", "dimensions": {"radius": 0.12, "height": 0.25}},
    {"id": "sensor", "name": "Sensor Pod", "category": "Research",
     "description": "Instrumentation package for environmental sampling.",
     "mass": 1.5, "reference_area": 0.01, "drag_coefficient": 1.0,
     "shape": "sphere", "dimensions": {"radius": 0.08}},
    {"id": "sample", "name": "Sample Canister", "category": "Research",
     "description": "Sealed canister for sample return.",
     "mass": 2.0, "reference_area": 0.015, "drag_coefficient": 0.90,
     "shape": "cylinder", "dimensions": {"radius": 0.06, "height": 0.18}},
    {"id": "dummy", "name": "Dummy Round", "category": "Training / Inert",
     "description": "Inert mass for drop training.",
     "mass": 1.0, "reference_area": 0.01, "drag_coefficient": 1.0,
     "shape": "sphere", "dimensions": {"radius": 0.06}},
    {"id": "practice", "name": "Practice Payload", "category": "Training / Inert",
     "description": "Generic practice shape for procedure training.",
     "mass": 1.2, "reference_area": 0.012, "drag_coefficient": 0.95,
     "shape": "box", "dimensions": {"length": 0.15, "width": 0.12, "height": 0.10}},
]

_state = {
    "selected_index": -1,
    "selected_category": None,
    "mass": None,
    "reference_area": None,
    "drag_coefficient": None,
    "dimensions_str": None,
}

_widget_refs = {"bc_ax": None, "fig": None, "payload_drop_ax": None, "_dropdown_timer": None}


def _get_archetype(index):
    if 0 <= index < len(PAYLOAD_LIBRARY):
        return PAYLOAD_LIBRARY[index]
    return None


def _payloads_for_category(cat):
    return [(i, p) for i, p in enumerate(PAYLOAD_LIBRARY) if p["category"] == cat]


def _dimensions_to_str(dims, shape):
    if not dims:
        return ""
    if shape == "box":
        return f"{dims.get('length', 0):.3f}, {dims.get('width', 0):.3f}, {dims.get('height', 0):.3f}"
    if shape == "sphere":
        return f"{dims.get('radius', 0):.3f}"
    if shape == "cylinder":
        return f"{dims.get('radius', 0):.3f}, {dims.get('height', 0):.3f}"
    return str(dims)


def _sync_state_from_archetype(index):
    p = _get_archetype(index)
    _state["selected_index"] = index
    if p is None:
        _state["mass"] = None
        _state["reference_area"] = None
        _state["drag_coefficient"] = None
        _state["dimensions_str"] = None
        return
    _state["mass"] = p["mass"]
    _state["reference_area"] = p["reference_area"]
    _state["drag_coefficient"] = p["drag_coefficient"]
    _state["dimensions_str"] = _dimensions_to_str(p.get("dimensions"), p.get("shape", "box"))


def get_payload_config():
    m = _state["mass"]
    a = _state["reference_area"]
    cd = _state["drag_coefficient"]
    try:
        bc = float(m) / (float(cd) * float(a)) if (m and a and cd) else None
    except (TypeError, ValueError, ZeroDivisionError):
        bc = None
    p = _get_archetype(_state["selected_index"])
    return {
        "name": p["name"] if p else None,
        "category": p["category"] if p else None,
        "mass": m,
        "reference_area": a,
        "drag_coefficient": cd,
        "dimensions_str": _state["dimensions_str"],
        "ballistic_coefficient": bc,
    }


def _update_bc_display():
    cfg = get_payload_config()
    bc = cfg["ballistic_coefficient"]
    fig = _widget_refs.get("fig")
    ax = _widget_refs.get("bc_ax")
    if ax is not None and fig is not None and ax in fig.axes:
        for t in list(ax.texts):
            t.remove()
        val = f"{bc:.4f}" if bc is not None else "—"
        ax.text(0.5, 0.5, val, transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color=_ACCENT, family="monospace")
        fig.canvas.draw_idle()


def render(ax, fig, interactive=True):
    ax.set_axis_off()
    ax.set_facecolor(_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.patch.set_facecolor(_BG)
    _widget_refs["fig"] = fig

    has_selection = _state["selected_index"] >= 0
    if has_selection and _state["mass"] is None:
        _sync_state_from_archetype(_state["selected_index"])

    # ----- Left: "Payload" label + button -> categories dropdown -> payloads dropdown -----
    left = ax.inset_axes([0.02, 0.48, 0.26, 0.50])
    left.set_facecolor(_PANEL)
    left.set_axis_off()
    left.set_xlim(0, 1)
    left.set_ylim(0, 1)
    left.add_patch(mpatches.Rectangle((0.02, 0.02), 0.96, 0.96, linewidth=1, edgecolor=_BORDER, facecolor="none", transform=left.transAxes))
    left.text(0.5, 0.96, "PAYLOAD LIBRARY", transform=left.transAxes, fontsize=9, color=_LABEL, ha="center", va="top", family="monospace")
    left.text(0.08, 0.78, "Payload", transform=left.transAxes, fontsize=10, color=_TEXT, va="center", family="monospace")

    # Main button: "Select payload" by default; click opens categories (level 1).
    btn_ax = fig.add_axes([0.08, 0.58, 0.22, 0.065])
    btn_ax.set_facecolor(_INPUT_BG)
    for s in btn_ax.spines.values():
        s.set_color(_BORDER)
    if has_selection:
        current_name = _get_archetype(_state["selected_index"])["name"]
        btn_label = current_name[:18] + "..." if len(current_name) > 18 else current_name
    else:
        btn_label = "Select payload"
    main_btn = Button(btn_ax, f"  {btn_label}  \u25BC  ", color=_INPUT_BG, hovercolor=_PANEL)
    main_btn.label.set_color(_TEXT)
    main_btn.label.set_fontsize(8)
    main_btn.label.set_fontfamily("monospace")

    def show_main_btn_text():
        if _state["selected_index"] >= 0:
            n = _get_archetype(_state["selected_index"])["name"]
            lbl = n[:18] + "..." if len(n) > 18 else n
        else:
            lbl = "Select payload"
        main_btn.label.set_text(f"  {lbl}  \u25BC  ")

    # ----- Center: visual + metadata (text stays inside; description truncated) -----
    vis = ax.inset_axes([0.30, 0.48, 0.26, 0.50])
    vis.set_facecolor(_PANEL)
    vis.set_axis_off()
    vis.set_xlim(0, 1)
    vis.set_ylim(0, 1)
    vis.set_clip_on(True)
    vis.add_patch(mpatches.Rectangle((0.03, 0.03), 0.94, 0.94, linewidth=1, edgecolor=_BORDER, facecolor="none", transform=vis.transAxes))
    p = _get_archetype(_state["selected_index"])
    if p:
        vis.add_patch(mpatches.Rectangle((0.15, 0.40), 0.70, 0.28, facecolor=_INPUT_BG, edgecolor=_ACCENT, linewidth=0.8))
        vis.text(0.5, 0.82, p["name"], transform=vis.transAxes, fontsize=10, ha="center", va="center", color=_ACCENT, family="monospace")
        vis.text(0.5, 0.70, p["category"], transform=vis.transAxes, fontsize=8, ha="center", va="center", color=_LABEL, family="monospace")
        desc = p["description"]
        if len(desc) > 38:
            desc = desc[:35].rsplit(" ", 1)[0] + "..."
        vis.text(0.5, 0.28, desc, transform=vis.transAxes, fontsize=7, ha="center", va="center", color=_TEXT, family="monospace")
    else:
        vis.text(0.5, 0.65, "1. Click dropdown below", transform=vis.transAxes, fontsize=9, ha="center", va="center", color=_LABEL, family="monospace")
        vis.text(0.5, 0.50, "2. Choose category", transform=vis.transAxes, fontsize=9, ha="center", va="center", color=_LABEL, family="monospace")
        vis.text(0.5, 0.35, "3. Choose payload", transform=vis.transAxes, fontsize=9, ha="center", va="center", color=_LABEL, family="monospace")

    # ----- Right: parameters (aligned, readable labels + fields) -----
    param = ax.inset_axes([0.58, 0.08, 0.40, 0.88])
    param.set_facecolor(_PANEL)
    param.set_axis_off()
    param.set_xlim(0, 1)
    param.set_ylim(0, 1)
    param.add_patch(mpatches.Rectangle((0.02, 0.02), 0.96, 0.96, linewidth=1, edgecolor=_BORDER, facecolor="none", transform=param.transAxes))
    param.text(0.5, 0.96, "PARAMETERS", transform=param.transAxes, fontsize=9, color=_LABEL, ha="center", va="top", family="monospace")

    row_h = 0.16
    y0 = 0.82
    labels_text = ["Mass (kg)", "Ref. area (m²)", "Drag coeff.", "Dimensions", "Ballistic coeff."]
    for k, y in enumerate([y0, y0 - row_h, y0 - 2*row_h, y0 - 3*row_h, y0 - 4*row_h]):
        param.text(0.06, y, labels_text[k], transform=param.transAxes, fontsize=8, color=_TEXT, va="center", family="monospace")

    param_bottom_fig = 0.06 + 0.08 * 0.78
    param_height_fig = 0.88 * 0.78
    row_centers = [param_bottom_fig + (y0 - j * row_h) * param_height_fig for j in range(5)]
    fleft, fwidth, fheight = 0.70, 0.20, 0.038
    mass_ax = fig.add_axes([fleft, row_centers[0] - fheight/2, fwidth, fheight])
    area_ax = fig.add_axes([fleft, row_centers[1] - fheight/2, fwidth, fheight])
    cd_ax = fig.add_axes([fleft, row_centers[2] - fheight/2, fwidth, fheight])
    dims_ax = fig.add_axes([fleft, row_centers[3] - fheight/2, fwidth, fheight])
    bc_ax = fig.add_axes([fleft, row_centers[4] - fheight/2, fwidth, fheight])

    for axx in (mass_ax, area_ax, cd_ax, dims_ax, bc_ax):
        axx.set_facecolor(_INPUT_BG)
        for s in axx.spines.values():
            s.set_color(_BORDER)
    bc_ax.set_axis_off()

    mass_tb = TextBox(mass_ax, "", initial=str(_state["mass"]) if _state["mass"] is not None else "—", textalignment="left")
    area_tb = TextBox(area_ax, "", initial=str(_state["reference_area"]) if _state["reference_area"] is not None else "—", textalignment="left")
    cd_tb = TextBox(cd_ax, "", initial=str(_state["drag_coefficient"]) if _state["drag_coefficient"] is not None else "—", textalignment="left")
    dims_tb = TextBox(dims_ax, "", initial=str(_state["dimensions_str"]) if _state["dimensions_str"] else "—", textalignment="left")

    _widget_refs["bc_ax"] = bc_ax
    _widget_refs["mass_tb"] = mass_tb
    _widget_refs["area_tb"] = area_tb
    _widget_refs["cd_tb"] = cd_tb
    _widget_refs["dims_tb"] = dims_tb

    def _refresh_param_boxes():
        fig = _widget_refs.get("fig")
        if fig is None:
            return
        for key, attr in [("mass_tb", "mass"), ("area_tb", "reference_area"), ("cd_tb", "drag_coefficient"), ("dims_tb", "dimensions_str")]:
            w = _widget_refs.get(key)
            if w is not None and hasattr(w, "ax") and w.ax in fig.axes:
                val = _state.get(attr)
                w.set_val(str(val) if val is not None else "—")

    def _parse_float(s, default=None):
        try:
            return float(s.strip()) if s else default
        except ValueError:
            return default

    def on_mass(t):
        v = _parse_float(t, _state["mass"])
        if v is not None:
            _state["mass"] = v
        _update_bc_display()

    def on_area(t):
        v = _parse_float(t, _state["reference_area"])
        if v is not None:
            _state["reference_area"] = v
        _update_bc_display()

    def on_cd(t):
        v = _parse_float(t, _state["drag_coefficient"])
        if v is not None:
            _state["drag_coefficient"] = v
        _update_bc_display()

    def on_dims(t):
        _state["dimensions_str"] = t.strip() if t else None
        _update_bc_display()

    if interactive:
        mass_tb.on_submit(on_mass)
        area_tb.on_submit(on_area)
        cd_tb.on_submit(on_cd)
        dims_tb.on_submit(on_dims)

    # Dropdown: categories + optional payloads sirf expanded parent ke neeche; toggle/main se band
    _category_axes = []
    _category_buttons = []
    _payload_axes = []
    _payload_buttons = []
    _showing = [None]
    _expanded_category = [None]

    def _clear_all_choice_buttons():
        for a in _payload_axes + _category_axes:
            if a in fig.axes:
                fig.delaxes(a)
        _payload_axes.clear()
        _payload_buttons.clear()
        _category_axes.clear()
        _category_buttons.clear()
        _showing[0] = None
        _expanded_category[0] = None

    _main_btn_bottom = 0.58
    _cat_btn_height = 0.052
    _cat_gap = 0.004
    _payload_btn_h = 0.036
    _payload_gap = 0.002
    _payload_indent = 0.018
    _payload_width = 0.202
    _gap_cat_to_payload = 0.006

    def _redraw_dropdown(expanded_label):
        """Dropdown ko poora redraw: har category ke baad sirf usi ke payloads (agar expanded ho)."""
        for a in _payload_axes + _category_axes:
            if a in fig.axes:
                fig.delaxes(a)
        _payload_axes.clear()
        _payload_buttons.clear()
        _category_axes.clear()
        _category_buttons.clear()

        expanded_idx = None
        if expanded_label is not None:
            for idx, c in enumerate(CATEGORIES):
                if c == expanded_label:
                    expanded_idx = idx
                    break

        y = _main_btn_bottom - _cat_gap
        for i, cat in enumerate(CATEGORIES):
            bottom = y - _cat_btn_height
            ax_c = fig.add_axes([0.08, bottom, 0.22, _cat_btn_height])
            ax_c.set_facecolor(_PANEL)
            for s in ax_c.spines.values():
                s.set_color(_ACCENT)
            is_exp = expanded_idx == i
            cat_label = "  " + cat + (" \u25BC" if is_exp else " \u25B6") + "  "
            b = Button(ax_c, cat_label, color=_PANEL, hovercolor=_INPUT_BG)
            b.label.set_color(_TEXT)
            b.label.set_fontsize(8)
            b.label.set_fontfamily("monospace")
            _category_axes.append(ax_c)
            _category_buttons.append(b)
            def _cat_cb(c, cat_idx):
                def _h(ev):
                    on_category_click(c, cat_idx)
                return _h
            b.on_clicked(_cat_cb(cat, i))
            y = bottom - _cat_gap

            if expanded_idx == i:
                payloads_in_cat = _payloads_for_category(cat)
                y -= _gap_cat_to_payload
                for idx, p in payloads_in_cat:
                    pname = p["name"]
                    y -= _payload_gap
                    p_bottom = y - _payload_btn_h
                    ax_p = fig.add_axes([0.08 + _payload_indent, p_bottom, _payload_width, _payload_btn_h])
                    ax_p.set_facecolor(_INPUT_BG)
                    for s in ax_p.spines.values():
                        s.set_color(_BORDER)
                    lbl = pname[:18] + ("..." if len(pname) > 18 else "")
                    bp = Button(ax_p, "  " + lbl + "  ", color=_INPUT_BG, hovercolor=_PANEL)
                    bp.label.set_color(_TEXT)
                    bp.label.set_fontsize(7)
                    bp.label.set_fontfamily("monospace")
                    _payload_axes.append(ax_p)
                    _payload_buttons.append(bp)
                    def _make_handler(ix):
                        def _h(ev):
                            _sync_state_from_archetype(ix)
                            _clear_all_choice_buttons()
                            show_main_btn_text()
                            _refresh_param_boxes()
                            _update_bc_display()
                            fig.canvas.draw()
                        return _h
                    bp.on_clicked(_make_handler(idx))
                    y = p_bottom - _payload_gap
                y -= _cat_gap

        _showing[0] = "payloads" if expanded_idx is not None else "categories"
        _expanded_category[0] = expanded_label
        fig.canvas.draw()

    def open_category_dropdown(event):
        if not interactive:
            # In Qt app, payload controls do not re-run the engine; keep honest.
            left.text(
                0.5,
                0.12,
                "LOCKED for this simulation snapshot (no engine re-run)",
                transform=left.transAxes,
                fontsize=7,
                color=_LABEL,
                ha="center",
                va="center",
                family="monospace",
            )
            fig.canvas.draw_idle()
            return
        if _showing[0] in ("categories", "payloads"):
            _clear_all_choice_buttons()
            fig.canvas.draw_idle()
            fig.canvas.draw()
            return
        _clear_all_choice_buttons()
        _redraw_dropdown(None)

    def on_category_click(label, category_index):
        def _apply():
            if _expanded_category[0] == label:
                _redraw_dropdown(None)
                return
            _state["selected_category"] = label
            _redraw_dropdown(label)

        # Defer redraw so it runs after the click handler (avoids backend re-entrancy).
        try:
            _timer = fig.canvas.new_timer(interval=50)
            _timer.single_shot = True
            _timer.add_callback(_apply)
            _widget_refs["_dropdown_timer"] = _timer
            _timer.start()
        except Exception:
            # Fallback: run immediately if timer not supported
            _apply()

    main_btn.on_clicked(open_category_dropdown)

    _update_bc_display()