# v2.7.0

**May 2026**

`dlordinal` **v2.7.0** includes the following updates:

---

### New Features
- **Unimodal output layers by ([Victor Vargas](https://github.com/victormvy)) in ([#148](https://github.com/ayrna/dlordinal/pull/148))**
  Added the binomial, Poisson, and Gaussian uncertainty output layers, together with a Gaussian uncertainty loss wrapper, expanding the library's unimodal modelling options.

- **Margin support for CDW-CE by ([Victor Vargas](https://github.com/victormvy)) in ([#150](https://github.com/ayrna/dlordinal/pull/150))**
  Added a configurable margin term to the CDW-CE loss to better control class separation in ordinal classification.

### Improvements & Maintenance
- **HCI test and dataset handling improvements by ([Victor Vargas](https://github.com/victormvy)) in ([#149](https://github.com/ayrna/dlordinal/pull/149)), ([#153](https://github.com/ayrna/dlordinal/pull/153)) and ([#154](https://github.com/ayrna/dlordinal/pull/154))**
  Fixed path-related issues in the HCI tests, reduced unnecessary dataset redownloads, added a download cache, improved retry back-off logic, and added offline test coverage for HCI and FGNet.

- **CI and GPU workflow maintenance by ([Victor Vargas](https://github.com/victormvy)) in ([#151](https://github.com/ayrna/dlordinal/pull/151)), ([#152](https://github.com/ayrna/dlordinal/pull/152)) and ([#156](https://github.com/ayrna/dlordinal/pull/156))**
  Introduced a global device fixture, added a `no_gpu_ci` pytest marker, and updated the GPU workflow to use the runner wrapper script and avoid OOM issues.
