"""
Payload Library tab.
DATA CATALOG for AIRDROP-X.
Strictly static data. No physics computation.
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
import matplotlib.patches as mpatches
from typing import Any, Dict, List, Optional

# Import unified military-grade theme
from product.ui.ui_theme import (
    BG_MAIN, BG_PANEL, BG_INPUT,
    TEXT_PRIMARY, TEXT_LABEL,
    ACCENT_GO,
    BORDER_SUBTLE,
    FONT_FAMILY, FONT_SIZE_BODY, FONT_SIZE_CAPTION, FONT_SIZE_H3,
    BUTTON_HOVER
)

CATEGORIES = [
    "Humanitarian / Relief",
    "Training / Inert",
    "Experimental / Research",
    "Commercial / Logistics",
    "Military / Tactical",
]

# =============================================================================
# Payload Library v1.0 — FROZEN (2026-02-10)
# =============================================================================
#
# PURPOSE:
# This library provides a standardized catalog of generic payloads for
# AIRDROP-X simulation and training scenarios.
#
# DISCLAIMERS:
# 1. ABSTRACT REPRESENTATION: All payloads are abstract geometric approximations
#    intended for aerodynamic simulation, not high-fidelity CAD models.
# 2. ASSUMED AERODYNAMICS: Drag Coefficients (Cd) are ASSUMED values based on
#    standard fluid dynamics ranges for the given geometry (e.g., Sphere Cd=0.5).
#    They are not computed from CFD or wind tunnel data.
# 3. NOT A REAL-WORLD REPLICA: Names and specifications are generic. Any
#    resemblance to specific real-world products or weapon systems is coincidental.
#
# STATUS:
# This version (v1.0) is FROZEN. No further changes to mass, geometry, or Cd
# parameters are permitted without a version increment and re-validation.
# =============================================================================

PAYLOAD_LIBRARY = [
    # --- Humanitarian / Relief ---
    {
        "id": "rel_sac_grain",
        "name": "Grain Sack",
        "category": "Humanitarian / Relief",
        "subcategory": "Consumables",
        "notes": "Woven polypropylene sack, shock tolerant.",
        "description": "Standard grain sack for air drop."
    },
    {
        "id": "rel_med_kit_s",
        "name": "Medical Kit",
        "category": "Humanitarian / Relief",
        "subcategory": "Medical",
        "notes": "Standard fast-aid kit.",
        "description": "Emergency medical supplies."
    },
    {
        "id": "rel_water_jerry",
        "name": "Water Jerrycan",
        "category": "Humanitarian / Relief",
        "subcategory": "Liquids",
        "notes": "Reinforced HDPE canister.",
        "description": "Water container for air drop."
    },
    {
        "id": "rel_blanket_roll",
        "name": "Thermal Blankets",
        "category": "Humanitarian / Relief",
        "subcategory": "Shelter",
        "notes": "Compressed wool blankets.",
        "description": "Thermal protection for refugees."
    },

    # --- Training / Inert ---
    {
        "id": "trg_cal_sphere",
        "name": "Calibration Sphere",
        "category": "Training / Inert",
        "subcategory": "Calibration",
        "notes": "Polished steel reference.",
        "description": "Precise aerodynamic reference shape."
    },
    {
        "id": "trg_dummy_box",
        "name": "Inert Training Load",
        "category": "Training / Inert",
        "subcategory": "Procedure",
        "notes": "Sand-filled ballast box.",
        "description": "Standard shape for procedure training."
    },
    {
        "id": "trg_sim_cyl",
        "name": "Simulated Canister",
        "category": "Training / Inert",
        "subcategory": "Procedure",
        "notes": "Concrete filled PVC pipe.",
        "description": "Low-cost simulation object."
    },

    # --- Experimental / Research ---
    {
        "id": "exp_atm_probe",
        "name": "Atmospheric Sonobuoy",
        "category": "Experimental / Research",
        "subcategory": "Sensors",
        "notes": "Deployable sensor array.",
        "description": "Atmospheric data collection unit."
    },
    {
        "id": "exp_reentry_test",
        "name": "Blunt Body Test Article",
        "category": "Experimental / Research",
        "subcategory": "Aerodynamics",
        "notes": "70 deg sphere-cone shape.",
        "description": "High-drag re-entry simulator."
    },
    {
        "id": "exp_bio_cont",
        "name": "Biological Sample Return",
        "category": "Experimental / Research",
        "subcategory": "Biological",
        "notes": "Impact hardened containment.",
        "description": "Secure biological sample container."
    },
    {
        "id": "exp_cubesat_sim",
        "name": "CubeSat Simulator",
        "category": "Experimental / Research",
        "subcategory": "Space Systems",
        "notes": "Standard 1U form factor dummy.",
        "description": "CubeSat form factor test unit."
    },

    # --- Commercial / Logistics ---
    {
        "id": "com_express_box",
        "name": "Express Parcel",
        "category": "Commercial / Logistics",
        "subcategory": "Delivery",
        "notes": "Standard cardboard shipping box.",
        "description": "Commercial delivery package."
    },
    {
        "id": "com_parts_bin",
        "name": "Spare Parts Bin",
        "category": "Commercial / Logistics",
        "subcategory": "Industrial",
        "notes": "Plastic tote with lid.",
        "description": "Industrial parts container."
    },
    {
        "id": "com_doc_tube",
        "name": "Map/Document Tube",
        "category": "Commercial / Logistics",
        "subcategory": "Documents",
        "notes": "Waterproof document container.",
        "description": "Secure document transport."
    },
    {
        "id": "com_cooler",
        "name": "Insulated Cooler",
        "category": "Commercial / Logistics",
        "subcategory": "Perishables",
        "notes": "Expanded polystyrene cooler.",
        "description": "Temperature controlled transport."
    },

    # --- Military / Tactical (Generic) ---
    {
        "id": "mil_smoke_can",
        "name": "Smoke Marker",
        "category": "Military / Tactical",
        "subcategory": "Signaling",
        "notes": "Generic signaling smoke canister.",
        "description": "Visual marker for LZ."
    },
    {
        "id": "mil_sensor_node",
        "name": "Remote Sensor Node",
        "category": "Military / Tactical",
        "subcategory": "ISR",
        "notes": "Ruggedized spherical sensor.",
        "description": "Ground sensor deployment."
    },
    {
        "id": "mil_ammo_box",
        "name": "Generic Ammo Can",
        "category": "Military / Tactical",
        "subcategory": "Resupply",
        "notes": "Steel container with ballast.",
        "description": "Standard ammunition container."
    },
    {
        "id": "mil_comms_droplink",
        "name": "Comms Relay Droplink",
        "category": "Military / Tactical",
        "subcategory": "Comms",
        "notes": "Self-righting communications buoy.",
        "description": "Tactical communications relay."
    }
]


def _dimensions_to_str(dims, shape):
    if not dims:
        return ""
    if shape == "box":
        return f"{dims.get('length', dims.get('length_m', 0)):.3f}, {dims.get('width', dims.get('width_m', 0)):.3f}, {dims.get('height', dims.get('height_m', 0)):.3f}"
    if shape == "sphere":
        return f"d={dims.get('diameter_m', dims.get('radius', 0)*2):.3f}"
    if shape in ("cylinder", "capsule"):
        return f"d={dims.get('diameter_m', dims.get('radius', 0)*2):.3f}, l={dims.get('length_m', dims.get('height', 0)):.3f}"
    return str(dims)


def _get_archetype(index):
    if 0 <= index < len(PAYLOAD_LIBRARY):
        return PAYLOAD_LIBRARY[index]
    return None


def _payloads_for_category(cat):
    # Match fuzzy to support legacy / slightly varied category names
    return [(i, p) for i, p in enumerate(PAYLOAD_LIBRARY) if p["category"].startswith(cat.split(" / ")[0])]



# ... (Imports preserved at top of file, but we need to re-add them if missing or just overwrite the class)

class PayloadLibraryTab:
    """Refactored PayloadLibraryTab for Dynamic Builder."""
    
    def __init__(self) -> None:
        self._state: Dict[str, Any] = {
            "selected_index": -1,
            "mass": None,
            "geometry_type": None, # "sphere", "cylinder", "box", etc.
            "dims": {},            # {"radius": 0.1, "length": 0.5, ...}
            "drag_coefficient": None,
            "calculated_area": None,
            "cd_uncertainty": None,
        }
        self._widget_refs: Dict[str, Any] = {
            "fig": None, 
            "mass_tb": None, 
            "cd_tb": None,
            "calc_area_txt": None,
            "dim_tbs": {} # Store per-dimension textboxes
        }
        
        # UI Lists
        self._category_axes = []
        self._category_buttons = []
        self._payload_axes = []
        self._payload_buttons = []
        self._geom_buttons = []
        self._geom_axes = []
        
        # Dropdown State
        self._showing = False
        self._expanded_category = None

    def _sync_state_from_archetype(self, index):
        """Loads identity only. Resets physical properties."""
        p = _get_archetype(index)
        self._state["selected_index"] = index
        # Reset physicals for new payload type
        self._state["mass"] = None 
        # Keep geometry/Cd if user wants to apply same shape to different payload? 
        # Requirement: "stepwise like user first what kind of payload... then select geometry"
        # So we probably reset geometry too to force flow.
        self._state["geometry_type"] = None
        self._state["dims"] = {}
        self._state["drag_coefficient"] = None
        self._state["calculated_area"] = None
        self._state["cd_uncertainty"] = None


    def _calculate_derived_physics(self):
        """Calculate Volume, Density, ref_area, etc."""
        m = self._state["mass"]
        g_type = self._state["geometry_type"]
        dims = self._state["dims"]
        
        vol = None
        area = None
        
        if g_type and dims:
            try:
                d: Dict[str, float] = {k: float(v) for k, v in dims.items() if v}
                if g_type == "sphere":
                    r = d.get("radius")
                    if r: 
                        area = 3.14159 * r * r
                        vol = (4/3) * 3.14159 * r**3
                elif g_type == "cylinder":
                    r = d.get("radius"); l = d.get("length")
                    if r and l: 
                        area = 2 * r * l
                        vol = 3.14159 * r**2 * l
                elif g_type == "box":
                    l = d.get("length"); w = d.get("width"); h = d.get("height")
                    if l and w and h: 
                        area = max(l*w, w*h, l*h)
                        vol = l * w * h
                elif g_type == "capsule":
                    r = d.get("radius"); l = d.get("length")
                    if r and l:
                        area = (2*r*l) + (3.14159 * r**2)
                        vol = (3.14159 * r**2 * l) + ((4/3) * 3.14159 * r**3)
                elif g_type == "blunt_cone":
                    r = d.get("radius"); l = d.get("length")
                    if r and l:
                        area = 3.14159 * r**2 
                        vol = (1/3) * 3.14159 * r**2 * l # Cone approx
            except Exception: pass
            
        density = (m / vol) if (m and vol) else None
        return area, vol, density

    def _update_calculations(self):
        """Auto-calculate Area and suggest Cd/Uncertainty."""
        area, vol, density = self._calculate_derived_physics()
        self._state["calculated_area"] = area
        self._state["calculated_volume"] = vol
        self._state["calculated_density"] = density
        
        g_type = self._state["geometry_type"]
        # Cd Suggestion (if not set)
        if self._state["drag_coefficient"] is None and g_type:
            defaults = {"sphere": 0.47, "cylinder": 0.90, "box": 1.15, "capsule": 0.50, "blunt_cone": 0.70}
            self._state["drag_coefficient"] = defaults.get(g_type)

        # Uncertainty
        u_rules = {"sphere": 0.05, "capsule": 0.10, "cylinder": 0.15, "blunt_cone": 0.15, "box": 0.20}
        self._state["cd_uncertainty"] = u_rules.get(g_type, 0.10)

        # UI Update
        self._refresh_param_display()

    def _save_config(self):
        """Save current config to custom_payloads.json"""
        import json, os
        cfg = self.get_payload_config()
        if not cfg["mass"]: return # Don't save empty
        
        fname = "custom_payloads.json"
        data = []
        if os.path.exists(fname):
            try:
                with open(fname, "r") as f: data = json.load(f)
            except: pass
        
        # Check if exists, update or append
        existing = next((i for i, d in enumerate(data) if d.get("name") == cfg["name"]), None)
        if existing is not None: data[existing] = cfg  # type: ignore[index]
        else: data.append(cfg)
        
        with open(fname, "w") as f: json.dump(data, f, indent=2)
        print(f"Saved payload: {cfg['name']}")

    def get_payload_config(self) -> Dict[str, Any]:
        m = self._state["mass"]
        a = self._state["calculated_area"]
        cd = self._state["drag_coefficient"]
        try:
            bc = float(m) / (float(cd) * float(a)) if (m and a and cd) else None
        except (TypeError, ValueError, ZeroDivisionError):
            bc = None
        
        p = _get_archetype(self._state["selected_index"])
        
        # Construct Geometry Dict for Validation
        g_type = self._state["geometry_type"]
        dims_map = self._state["dims"]
        geometry_data: Dict[str, Any] = {"type": g_type, "dimensions": {}}
        
        # Map flat UI dims back to validation schema (radius -> diameter, etc)
        # Validation expects: diameter_m, length_m, width_m, height_m
        if g_type == "sphere":
            if "radius" in dims_map: geometry_data["dimensions"]["diameter_m"] = float(dims_map["radius"]) * 2
        elif g_type in ("cylinder", "capsule"): # validation expects radius? No, validation.py checks diameter_m usually.
             # Actually existing validation uses: sphere(diameter_m), cylinder(diameter_m, length_m), box(length_m, width_m, height_m)
             # Let's conform.
             if "radius" in dims_map: geometry_data["dimensions"]["diameter_m"] = float(dims_map["radius"]) * 2
             if "length" in dims_map: geometry_data["dimensions"]["length_m"] = float(dims_map["length"])
        elif g_type == "box":
             if "length" in dims_map: geometry_data["dimensions"]["length_m"] = float(dims_map["length"])
             if "width" in dims_map: geometry_data["dimensions"]["width_m"] = float(dims_map["width"])
             if "height" in dims_map: geometry_data["dimensions"]["height_m"] = float(dims_map["height"])
        elif g_type == "blunt_cone":
             if "radius" in dims_map: geometry_data["dimensions"]["base_diameter_m"] = float(dims_map["radius"]) * 2
             if "length" in dims_map: geometry_data["dimensions"]["length_m"] = float(dims_map["length"])

        # Validate
        from product.payloads.geometry_validation import validate_geometry, validate_aerodynamics
        if g_type:
            try:
                validate_geometry(g_type, geometry_data["dimensions"])
                if cd is not None:
                    validate_aerodynamics(g_type, cd)
            except ValueError as e:
                print(f"Validation Warning: {e}")

        return {
            "name": p["name"] if p else "Custom Payload",
            "category": p["category"] if p else "Custom",
            "mass": m,
            "reference_area": a,
            "drag_coefficient": cd,
            "ballistic_coefficient": bc,
            "geometry": geometry_data 
        }

    def _refresh_param_display(self):
        fig = self._widget_refs.get("fig")
        if not fig: return
        
        # Mass
        tb = self._widget_refs.get("mass_tb")
        if tb: tb.set_val(str(self._state["mass"]) if self._state["mass"] is not None else "")
        
        # Dimensions (Rebuild if needed? No, just keep values)
        # Area
        txt = self._widget_refs.get("calc_area_txt")
        if txt:
            val = self._state["calculated_area"]
            txt.set_text(f"{val:.4f} m²" if val else "—")
            
        # Cd
        tb = self._widget_refs.get("cd_tb")
        if tb: tb.set_val(str(self._state["drag_coefficient"]) if self._state["drag_coefficient"] else "")

        # BC
        self._update_bc_display()
        fig.canvas.draw_idle()

    def _update_bc_display(self):
        cfg = self.get_payload_config()
        bc = cfg["ballistic_coefficient"]
        ax = self._widget_refs.get("bc_ax")
        if ax is not None:
            for t in list(ax.texts): t.remove()
            val = f"{bc:.2f}" if bc is not None else "—"
            ax.text(0.5, 0.5, val, transform=ax.transAxes, ha="center", va="center", fontsize=14, color=ACCENT_GO, family="monospace", weight='bold')

    def _clear_all_choice_buttons(self, fig):
        for a in self._payload_axes + self._category_axes:
            if a in fig.axes: fig.delaxes(a)
        self._payload_axes.clear()
        self._payload_buttons.clear()
        self._category_axes.clear()
        self._category_buttons.clear()
        self._showing = False
        self._expanded_category = None

    def _redraw_dropdown(self, fig, main_btn, expanded_label, on_category_click_cb):
        self._clear_all_choice_buttons(fig)
        
        expanded_idx = None
        if expanded_label is not None:
            for idx, c in enumerate(CATEGORIES):
                if c == expanded_label: 
                    expanded_idx = idx; break
        
        # Position dropdown below the button (button is at ~0.82 in figure coords)
        y = 0.78
        for i, cat in enumerate(CATEGORIES):
            bottom = y - 0.048
            ax_c = fig.add_axes([0.04, bottom, 0.22, 0.048])
            ax_c.set_facecolor(BG_PANEL)
            for s in ax_c.spines.values(): s.set_color(ACCENT_GO)
            is_exp = expanded_idx == i
            cat_label = "  " + cat.split(" / ")[0] + (" \u25BC" if is_exp else " \u25B6") + "  "
            b = Button(ax_c, cat_label, color=BG_PANEL, hovercolor=BG_INPUT)
            b.label.set_color(TEXT_PRIMARY); b.label.set_fontsize(8); b.label.set_fontfamily("monospace")
            self._category_axes.append(ax_c); self._category_buttons.append(b)
            
            def _c_cb(c_l, c_i):
                return lambda ev: on_category_click_cb(c_l, c_i)
            b.on_clicked(_c_cb(cat, i))
            y = bottom - 0.004
            
            if is_exp:
                payloads_in_cat = _payloads_for_category(cat)
                y -= 0.004
                for idx, p in payloads_in_cat:
                    pname = p["name"]
                    y -= 0.002
                    p_bottom = y - 0.034
                    ax_p = fig.add_axes([0.06, p_bottom, 0.20, 0.034])
                    ax_p.set_facecolor(BG_INPUT)
                    for s in ax_p.spines.values(): s.set_color(BORDER_SUBTLE)
                    lbl = pname[:20] + ".." if len(pname)>20 else pname
                    bp = Button(ax_p, "  " + lbl + "  ", color=BG_INPUT, hovercolor=BG_PANEL)
                    bp.label.set_color(TEXT_PRIMARY); bp.label.set_fontsize(7); bp.label.set_fontfamily("monospace")
                    self._payload_axes.append(ax_p); self._payload_buttons.append(bp)
                    
                    def _p_cb(ix):
                        def _h(ev):
                            self._sync_state_from_archetype(ix)
                            self._clear_all_choice_buttons(fig)
                            main_btn.label.set_text(f"  {_get_archetype(ix)['name']}  \u25BC  ")
                            # Reset UI for Step 2 & 3
                            self._rebuild_geometry_ui(fig)
                        return _h
                    bp.on_clicked(_p_cb(idx))
                    y = p_bottom - 0.002
                y -= 0.004
        
        self._showing = True
        fig.canvas.draw()

    def _rebuild_geometry_ui(self, fig):
        # Clear existing geometry specific widgets
        for ax in self._geom_axes:
            if ax in fig.axes: fig.delaxes(ax)
        self._geom_axes.clear()
        self._geom_buttons.clear()
        self._widget_refs["dim_tbs"] = {}  # Reset dimension textbox refs
        
        # Right panel content area (panel bg at [0.58, 0.08, 0.40, 0.88])
        px = 0.60       # left edge of content inside panel
        pw = 0.36       # usable width
        row_h = 0.038   # input row height
        gap = 0.012     # vertical gap between rows
        sgap = 0.022    # gap between sections
        
        y = 0.87  # start near top of panel
        
        # ── 1. MASS INPUT ────────────────────────────────────────────────
        ax_lbl = fig.add_axes([px, y, 0.10, row_h])
        ax_lbl.set_axis_off()
        ax_lbl.text(0, 0.5, "Mass (kg):", va="center", color=TEXT_LABEL, fontsize=8, family="monospace")
        self._geom_axes.append(ax_lbl)
        
        ax_mass = fig.add_axes([px + 0.12, y, 0.14, row_h])
        ax_mass.set_facecolor(BG_INPUT)
        for s in ax_mass.spines.values(): s.set_color(ACCENT_GO)
        init_mass = str(self._state["mass"]) if self._state["mass"] is not None else ""
        tb_mass = TextBox(ax_mass, "", initial=init_mass, textalignment="center")
        tb_mass.label.set_color(TEXT_PRIMARY)
        def _on_mass(v):
            try: self._state["mass"] = float(v); self._update_calculations(); self._refresh_param_display()
            except Exception: pass
        tb_mass.on_submit(_on_mass)
        self._widget_refs["mass_tb"] = tb_mass
        self._geom_axes.append(ax_mass)
        
        # ── 2. SHAPE SELECTOR ────────────────────────────────────────────
        y -= (row_h + sgap)
        ax_lbl2 = fig.add_axes([px, y, 0.10, row_h])
        ax_lbl2.set_axis_off()
        ax_lbl2.text(0, 0.5, "Shape:", va="center", color=TEXT_LABEL, fontsize=8, family="monospace")
        self._geom_axes.append(ax_lbl2)
        
        shapes = ["sphere", "cylinder", "box", "capsule", "blunt_cone"]
        btn_w, btn_h, btn_gap = 0.065, 0.030, 0.004
        curr_shape = self._state["geometry_type"]
        
        y -= (row_h + 0.005)
        for i, sh in enumerate(shapes):
            row, col = i // 3, i % 3
            sx = px + col * (btn_w + btn_gap)
            sy = y - row * (btn_h + btn_gap)
            
            ax_s = fig.add_axes([sx, sy, btn_w, btn_h])
            is_sel = curr_shape == sh
            bg = ACCENT_GO if is_sel else BG_INPUT
            lbl = sh.replace("_", " ").title()
            if len(lbl) > 7: lbl = lbl[:6] + "."
            bs = Button(ax_s, lbl, color=bg, hovercolor=ACCENT_GO)
            bs.label.set_fontsize(6); bs.label.set_fontfamily("monospace")
            bs.label.set_color("black" if is_sel else TEXT_PRIMARY)
            
            def _mk_sh(s):
                def _h(ev):
                    self._state["geometry_type"] = s
                    self._state["dims"] = {}
                    self._state["drag_coefficient"] = None
                    self._update_calculations()
                    self._rebuild_geometry_ui(fig)
                    fig.canvas.draw()
                return _h
            bs.on_clicked(_mk_sh(sh))
            self._geom_buttons.append(bs)
            self._geom_axes.append(ax_s)
        
        n_rows = (len(shapes) + 2) // 3
        y -= n_rows * (btn_h + btn_gap)
        
        # ── 3. DIMENSION INPUTS ──────────────────────────────────────────
        y -= sgap
        if curr_shape:
            ax_lbl3 = fig.add_axes([px, y, pw, row_h])
            ax_lbl3.set_axis_off()
            ax_lbl3.text(0, 0.5, f"Dimensions ({curr_shape}):", va="center", color=TEXT_LABEL, fontsize=8, family="monospace")
            self._geom_axes.append(ax_lbl3)
            
            req_dims = []
            if curr_shape == "sphere": req_dims = ["radius"]
            elif curr_shape in ["cylinder", "capsule", "blunt_cone"]: req_dims = ["radius", "length"]
            elif curr_shape == "box": req_dims = ["length", "width", "height"]
            
            for dim_name in req_dims:
                y -= (row_h + gap)
                ax_d_lbl = fig.add_axes([px + 0.02, y, 0.08, row_h])
                ax_d_lbl.set_axis_off()
                ax_d_lbl.text(1, 0.5, f"{dim_name.title()} (m):", ha="right", va="center", color=TEXT_PRIMARY, fontsize=7, family="monospace")
                self._geom_axes.append(ax_d_lbl)
                
                ax_d = fig.add_axes([px + 0.12, y, 0.14, row_h])
                ax_d.set_facecolor(BG_INPUT)
                for s in ax_d.spines.values(): s.set_color(BORDER_SUBTLE)
                
                val = self._state["dims"].get(dim_name, "")
                tb_d = TextBox(ax_d, "", initial=str(val), textalignment="center")
                tb_d.label.set_color(TEXT_PRIMARY)
                
                def _mk_dim(dn):
                    def _h(v):
                        try:
                            self._state["dims"][dn] = float(v)
                            self._update_calculations()
                            self._refresh_param_display()
                        except Exception: pass
                    return _h
                tb_d.on_submit(_mk_dim(dim_name))
                # CRITICAL: Store ref to prevent garbage collection
                self._widget_refs["dim_tbs"][dim_name] = tb_d
                self._geom_axes.append(ax_d)
        
        # ── 4. AERODYNAMICS ──────────────────────────────────────────────
        y -= (row_h + sgap)
        
        # Reference Area (read-only)
        ax_area = fig.add_axes([px, y, pw, row_h])
        ax_area.set_axis_off()
        val_area = self._state["calculated_area"]
        txt = f"Ref Area: {val_area:.4f} m²" if val_area else "Ref Area: —"
        ax_area.text(0, 0.5, txt, va="center", color=ACCENT_GO, fontsize=8, family="monospace")
        self._widget_refs["calc_area_txt"] = ax_area.texts[0]
        self._geom_axes.append(ax_area)
        
        # Cd Input
        y -= (row_h + gap)
        ax_cd_lbl = fig.add_axes([px, y, 0.10, row_h])
        ax_cd_lbl.set_axis_off()
        ax_cd_lbl.text(0, 0.5, "Cd:", va="center", color=TEXT_LABEL, fontsize=8, family="monospace")
        self._geom_axes.append(ax_cd_lbl)
        
        ax_cd = fig.add_axes([px + 0.12, y, 0.14, row_h])
        ax_cd.set_facecolor(BG_INPUT)
        for s in ax_cd.spines.values(): s.set_color(ACCENT_GO)
        init_cd = str(self._state["drag_coefficient"]) if self._state["drag_coefficient"] else ""
        tb_cd = TextBox(ax_cd, "", initial=init_cd, textalignment="center")
        def _on_cd(v):
            try: self._state["drag_coefficient"] = float(v); self._update_bc_display()
            except Exception: pass
        tb_cd.on_submit(_on_cd)
        self._widget_refs["cd_tb"] = tb_cd
        self._geom_axes.append(ax_cd)

        # Uncertainty
        unc = self._state["cd_uncertainty"]
        if unc:
            ax_u = fig.add_axes([px + 0.28, y, 0.08, row_h])
            ax_u.set_axis_off()
            ax_u.text(0, 0.5, f"±{unc}", va="center", color="gray", fontsize=7, family="monospace")
            self._geom_axes.append(ax_u)
            
        # BC Display at bottom
        self._update_bc_display()

    def render(self, ax, fig, interactive=True, run_simulation_callback=None):
        ax.set_axis_off()
        ax.set_facecolor(BG_MAIN); fig.patch.set_facecolor(BG_MAIN)
        self._widget_refs["fig"] = fig
        self._widget_refs["_buttons"] = []  # Keep button refs alive
        
        has_sel = self._state["selected_index"] >= 0

        # =====================================================================
        # LAYOUT — Clean 3-column grid, all panels share same top/bottom
        #   COL-1 (LEFT):   [0.02, 0.15, 0.26, 0.82]  Identity + Dropdown
        #   COL-2 (CENTER): top [0.30, 0.50, 0.26, 0.47]  Analysis
        #                   bot [0.30, 0.15, 0.26, 0.33]  Ballistic Coeff
        #   COL-3 (RIGHT):  [0.58, 0.15, 0.40, 0.82]  Physics Parameters
        # =====================================================================

        top_y = 0.15   # shared bottom for all panels
        top_h = 0.82   # shared height for full panels

        # ── LEFT: Identity ───────────────────────────────────────────────
        left = ax.inset_axes([0.02, top_y, 0.26, top_h])
        left.set_facecolor(BG_PANEL); left.set_axis_off()
        left.add_patch(mpatches.Rectangle((0,0),1,1, linewidth=1, edgecolor=BORDER_SUBTLE, facecolor="none", transform=left.transAxes))
        left.text(0.5, 0.97, "STEP 1: IDENTITY", transform=left.transAxes, fontsize=8, color=TEXT_LABEL, ha="center", va="top", family="monospace")
        
        # Dropdown button (figure coords, inside left panel)
        btn_ax = fig.add_axes([0.04, 0.84, 0.22, 0.05])
        btn_ax.set_facecolor(BG_INPUT)
        curr = _get_archetype(self._state["selected_index"])["name"] if has_sel else "Select Payload..."
        main_btn = Button(btn_ax, f"  {curr}  \u25BC", color=BG_INPUT, hovercolor=BG_PANEL)
        main_btn.label.set_color(TEXT_PRIMARY); main_btn.label.set_fontsize(8); main_btn.label.set_fontfamily("monospace")
        self._widget_refs["_buttons"].append(main_btn)
        
        def _toggle_dd(ev):
            if not interactive: return
            if self._showing: 
                self._clear_all_choice_buttons(fig); fig.canvas.draw()
            else: 
                self._redraw_dropdown(fig, main_btn, None, _dd_cat_clk)

        def _dd_cat_clk(lbl, idx):
             self._state["selected_category"] = lbl
             self._redraw_dropdown(fig, main_btn, lbl, _dd_cat_clk)

        main_btn.on_clicked(_toggle_dd)

        # Selection info
        if has_sel:
             p = _get_archetype(self._state["selected_index"])
             left.text(0.5, 0.28, p["notes"], transform=left.transAxes, ha="center", fontsize=7, color=TEXT_PRIMARY, wrap=True, family="monospace")
             left.text(0.5, 0.12, p.get("category", ""), transform=left.transAxes, ha="center", fontsize=6, color=ACCENT_GO, family="monospace")
             left.text(0.5, 0.05, p.get("subcategory", ""), transform=left.transAxes, ha="center", fontsize=6, color="gray", family="monospace")

        # ── CENTER TOP: Analysis ─────────────────────────────────────────
        vis = ax.inset_axes([0.30, 0.50, 0.26, 0.47])
        vis.set_facecolor(BG_PANEL); vis.set_axis_off()
        vis.add_patch(mpatches.Rectangle((0,0),1,1, linewidth=1, edgecolor=BORDER_SUBTLE, facecolor="none", transform=vis.transAxes))
        vis.text(0.5, 0.95, "ANALYSIS", transform=vis.transAxes, fontsize=8, color=TEXT_LABEL, ha="center", family="monospace")
        
        if has_sel:
            p = _get_archetype(self._state["selected_index"])
            vis.text(0.5, 0.78, p["description"], transform=vis.transAxes, ha="center", va="center", fontsize=7, color=TEXT_PRIMARY, wrap=True)
            
            warn_y = 0.55
            rho = self._state.get("calculated_density")
            if rho:
                if rho < 10: 
                    vis.text(0.5, warn_y, "⚠ Low Density (<10 kg/m³)", color="orange", ha="center", fontsize=7); warn_y -= 0.12
                elif rho > 20000:
                    vis.text(0.5, warn_y, "⚠ High Density (>20k kg/m³)", color="red", ha="center", fontsize=7); warn_y -= 0.12
                else:
                    vis.text(0.5, warn_y, f"Density: {rho:.1f} kg/m³ ✓", color="gray", ha="center", fontsize=7); warn_y -= 0.12
            
            cfg = self.get_payload_config()
            bc = cfg["ballistic_coefficient"]
            if bc is not None and float(bc) > 1000:
                 vis.text(0.5, warn_y, "⚠ Kinetic Penetrator (BC > 1000)", color="red", ha="center", fontsize=7)
        else:
            vis.text(0.5, 0.50, "No payload selected", transform=vis.transAxes, ha="center", color="gray", fontsize=8)

        # ── RIGHT: Physics Parameters ────────────────────────────────────
        param_bg = ax.inset_axes([0.58, top_y, 0.40, top_h])
        param_bg.set_facecolor(BG_PANEL); param_bg.set_axis_off()
        param_bg.add_patch(mpatches.Rectangle((0,0),1,1, linewidth=1, edgecolor=BORDER_SUBTLE, facecolor="none", transform=param_bg.transAxes))
        param_bg.text(0.5, 0.97, "STEP 2 & 3: PHYSICS", transform=param_bg.transAxes, fontsize=8, color=TEXT_LABEL, ha="center", va="top", family="monospace")
        
        # ── CENTER BOTTOM: Ballistic Coefficient ─────────────────────────
        bc_bg = ax.inset_axes([0.30, top_y, 0.26, 0.33])
        bc_bg.set_facecolor(BG_PANEL); bc_bg.set_axis_off()
        bc_bg.add_patch(mpatches.Rectangle((0,0),1,1, linewidth=1, edgecolor=BORDER_SUBTLE, facecolor="none", transform=bc_bg.transAxes))
        bc_bg.text(0.5, 0.92, "BALLISTIC COEFF", transform=bc_bg.transAxes, fontsize=8, color=TEXT_LABEL, ha="center", family="monospace")
        
        # BC Value axis
        ax_bc_val = fig.add_axes([0.32, 0.22, 0.22, 0.18])
        ax_bc_val.set_axis_off()
        self._widget_refs["bc_ax"] = ax_bc_val

        # ── Bottom Buttons (Save / Run Sim) ──────────────────────────────
        if interactive:
            ax_save = fig.add_axes([0.31, 0.16, 0.11, 0.04])
            btn_save = Button(ax_save, "Save", color=BG_INPUT, hovercolor=ACCENT_GO)
            btn_save.label.set_color(TEXT_PRIMARY); btn_save.label.set_fontsize(7); btn_save.label.set_fontfamily("monospace")
            btn_save.on_clicked(lambda ev: self._save_config())
            self._widget_refs["_buttons"].append(btn_save)
            
            ax_run = fig.add_axes([0.43, 0.16, 0.12, 0.04])
            col_run = "#005500" if run_simulation_callback else "#333333"
            btn_run = Button(ax_run, "RUN SIM", color=col_run, hovercolor=ACCENT_GO)
            btn_run.label.set_color("white"); btn_run.label.set_fontsize(7); btn_run.label.set_weight("bold"); btn_run.label.set_fontfamily("monospace")
            if run_simulation_callback:
                btn_run.on_clicked(lambda ev: run_simulation_callback(self.get_payload_config()))
            self._widget_refs["_buttons"].append(btn_run)

        # Build dynamic geometry UI or show placeholder
        if has_sel:
            self._rebuild_geometry_ui(fig)
        else:
            ax_ph = fig.add_axes([0.62, 0.48, 0.34, 0.08])
            ax_ph.set_axis_off()
            ax_ph.text(0.5, 0.5, "Select a payload identity to configure.", ha="center", color="gray", fontsize=8, family="monospace")
            self._geom_axes.append(ax_ph)

# Global singleton instance for backward compatibility with `render` shim
_tab_instance = PayloadLibraryTab()

def render(ax, fig, interactive=True, run_simulation_callback=None):
    """Backwards-compatible render function using the global singleton."""
    try:
        _tab_instance.render(ax, fig, interactive, run_simulation_callback)
    except Exception as e:
        print(f"UI RENDER ERROR: {e}")
        import traceback; traceback.print_exc()

def get_payload_config():
    """Backwards-compatible accessor."""
    return _tab_instance.get_payload_config()
