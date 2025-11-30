# Quick Start Guide

## Your Site is Configured! ✅

Your al-folio academic website has been set up with:
- **Name:** Rishi Malhotra
- **Title:** ML Engineer @ Apple
- **Bio:** Working on transformer-based RL, LoRA training, and exploring diffusion models
- **Math rendering:** Enabled (MathJax 3)
- **Example post:** Introduction to Diffusion Models with LaTeX equations

## Get Your Site Running (Choose One)

### Option 1: Fix Ruby & Run Locally (Recommended)

Run the setup script to install a working Ruby environment:

```bash
./setup.sh
```

This will:
1. Install rbenv and Ruby 3.1.0 with OpenSSL support
2. Install all Jekyll dependencies
3. Get your site ready to run

Then start the server:
```bash
bundle exec jekyll serve
```

Visit: **http://localhost:4000**

### Option 2: Use Docker

Start Docker Desktop, then:

```bash
docker-compose up
```

Visit: **http://localhost:8080**

## Current Issue: Ruby OpenSSL

Your current Ruby installation (3.0.0 via RVM) doesn't have OpenSSL support compiled in. This is a common issue on macOS.

**The `setup.sh` script fixes this by installing Ruby 3.1.0 via rbenv with proper OpenSSL support.**

## What's Been Done

✅ Configured `_config.yml` with your personal information
✅ Enabled MathJax for LaTeX rendering
✅ Created example blog post: `_posts/2025-11-27-introduction-to-diffusion-models.md`
✅ Set up GitHub Pages deployment configuration
✅ Created comprehensive documentation (`GETTING_STARTED.md`)

## What's Next

1. **Run the setup script** to fix Ruby: `./setup.sh`
2. **Start the local server** to preview your site
3. **Customize your about page**: Edit `_pages/about.md`
4. **Add your photo**: Place it in `assets/img/prof_pic.jpg`
5. **Write your first post**: See `GETTING_STARTED.md` for post creation guide
6. **Deploy to GitHub Pages**: See deployment section in `GETTING_STARTED.md`

## Need Help?

- **Full documentation**: See `GETTING_STARTED.md`
- **Example post with LaTeX**: `_posts/2025-11-27-introduction-to-diffusion-models.md`
- **al-folio docs**: https://github.com/alshedivat/al-folio

## Quick Commands Reference

```bash
# Setup (one-time)
./setup.sh

# Run site locally
bundle exec jekyll serve

# Run with Docker
docker-compose up

# Create new post
touch _posts/2025-11-27-your-post-title.md

# Deploy to GitHub Pages
git add .
git commit -m "Update site"
git push
```

## Deployment to GitHub Pages

1. Create a repo named `yourusername.github.io`
2. Update `url` in `_config.yml` to `https://yourusername.github.io`
3. Push your code:
   ```bash
   git remote add origin https://github.com/yourusername/yourusername.github.io.git
   git push -u origin main
   ```
4. Enable GitHub Pages in repo Settings → Pages
5. Your site will be live at `https://yourusername.github.io`

**Full deployment guide in `GETTING_STARTED.md`**

---

**Ready to start?** Run `./setup.sh` to get your local environment working!
