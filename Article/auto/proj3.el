(TeX-add-style-hook
 "proj3"
 (lambda ()
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "aa"
    "aa10"
    "graphicx"
    "txfonts"
    "amsmath"
    "mathrsfs"
    "booktabs"
    "tikz"
    "hyperref")
   (LaTeX-add-labels
    "eq:setup1"
    "eq:setup2"
    "eq:r_mass"
    "eq:r_R"
    "eq:r_t"
    "eq:r_v"
    "eq:r_a"
    "eq:collisions"
    "sec:sun_earth_mars"
    "fig:sun_earth_mars"
    "fig:sun_earth_mars_energy"
    "sec:sun_earth_moon"
    "fig:sun_earth_moon"
    "fig:sun_earth_moon_energy"
    "fig:lagrange_points"
    "tab:params"
    "sec:l1"
    "eq:l1"
    "eq:hill_l1"
    "fig:l1_unstable"
    "fig:l1_stable"
    "fig:l1_relative_r"
    "fig:l5_orb"
    "fig:l5_radial_perturbation"
    "fig:l5_perturbed_orb"
    "fig:l5_z"
    "fig:l5_jup")
   (LaTeX-add-bibliographies
    "nbody"))
 :latex)

