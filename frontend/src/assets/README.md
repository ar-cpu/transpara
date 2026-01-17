# Assets Folder

## Logo Setup

The Transpara frontend expects a logo file at:
```
frontend/src/assets/logo.jpg
```

### To add your logo:

1. Place your logo image file in this folder
2. Name it `logo.jpg` (or update the reference in `app.component.html` line 12)
3. Supported formats: JPG, PNG, SVG
4. Recommended size: 300x300 pixels or larger (will auto-scale to fit)

### Current Behavior:

- **If logo.jpg exists:** Displays your logo in the header
- **If logo.jpg is missing:** Shows a fallback icon (layer-group icon in dark circle)

The app handles the logo error gracefully with the `logoLoadError` signal in `app.component.ts`.

### Example:

```bash
# Copy your logo to assets folder
copy your-logo.jpg frontend\src\assets\logo.jpg
```

Then restart the Angular dev server if it's running:
```bash
npm start
```
