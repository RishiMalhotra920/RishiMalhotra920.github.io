# Getting Started with Your al-folio Academic Website

This guide will help you set up and run your personal academic website using the al-folio Jekyll theme.

## Prerequisites

- Git
- Either Docker OR Ruby 2.7+ with OpenSSL support
- A text editor

## Quick Start

### Option 1: Using Docker (Recommended)

Docker provides the easiest setup with no Ruby installation required.

1. **Start Docker Desktop** (if not already running)

2. **Start the development server:**
   ```bash
   docker-compose up
   ```

3. **View your site:**
   Open your browser to `http://localhost:8080`

4. **Stop the server:**
   Press `Ctrl+C` or run:
   ```bash
   docker-compose down
   ```

### Option 2: Native Ruby Setup

If you prefer running Jekyll natively:

#### Fix Ruby/OpenSSL Issues (macOS)

If you encounter OpenSSL errors, you'll need to reinstall Ruby with OpenSSL support:

1. **Install rbenv and ruby-build:**
   ```bash
   brew install rbenv ruby-build
   ```

2. **Install Ruby with OpenSSL:**
   ```bash
   rbenv install 3.1.0
   rbenv global 3.1.0
   ```

3. **Add to your shell profile** (`~/.zshrc` or `~/.bash_profile`):
   ```bash
   eval "$(rbenv init - zsh)"
   ```

4. **Restart your terminal** and verify:
   ```bash
   ruby --version  # should show 3.1.0
   ```

#### Install Dependencies

Once Ruby is properly configured:

```bash
bundle install
```

#### Run the Development Server

```bash
bundle exec jekyll serve
```

Your site will be available at `http://localhost:4000`

## Creating New Blog Posts

### File Naming Convention

Blog posts must follow this naming pattern:
```
_posts/YYYY-MM-DD-title-with-hyphens.md
```

Example:
```
_posts/2025-11-27-my-awesome-post.md
```

### Basic Post Template

Create a new file in `_posts/` with the following front matter:

```markdown
---
layout: post
title: Your Post Title
date: 2025-11-27 10:00:00-0800
description: A brief description of your post
tags: machine-learning transformers research
categories: ml-research
related_posts: false
---

Your content goes here...
```

### Front Matter Options

- `title`: The post title (required)
- `date`: Publication date in `YYYY-MM-DD HH:MM:SS-TIMEZONE` format (required)
- `description`: Brief description for SEO and previews
- `tags`: Space or comma-separated tags
- `categories`: Post categories
- `related_posts`: Set to `true` to show related posts at the bottom

## Writing LaTeX Math

The site uses MathJax 3 for beautiful math rendering.

### Inline Math

Wrap math in `$$ ... $$` within a paragraph:

```markdown
The equation $$ E = mc^2 $$ shows the relationship between energy and mass.
```

### Display Math

Use `$$ ... $$` on separate lines:

```markdown
The quadratic formula is:

$$
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
$$
```

### Numbered Equations

Use `\begin{equation}...\end{equation}` with labels:

```markdown
\begin{equation}
\label{eq:important}
f(x) = \int_{-\infty}^{\infty} e^{-x^2} dx
\end{equation}

As shown in equation \eqref{eq:important}, we can see...
```

### Multi-line Equations

Use the `align` environment:

```markdown
\begin{align}
\nabla \times \mathbf{E} &= -\frac{\partial \mathbf{B}}{\partial t} \\
\nabla \times \mathbf{H} &= \mathbf{J} + \frac{\partial \mathbf{D}}{\partial t}
\end{align}
```

### Common LaTeX Commands

- Vectors: `\mathbf{x}` → $$ \mathbf{x} $$
- Greek letters: `\alpha`, `\beta`, `\theta` → $$ \alpha, \beta, \theta $$
- Summation: `\sum_{i=1}^{n}` → $$ \sum_{i=1}^{n} $$
- Integral: `\int_{a}^{b}` → $$ \int_{a}^{b} $$
- Fraction: `\frac{a}{b}` → $$ \frac{a}{b} $$
- Subscript/Superscript: `x_i`, `x^2` → $$ x_i, x^2 $$

## Deploying to GitHub Pages

### Initial Setup

1. **Create a GitHub repository** named `yourusername.github.io` (replace `yourusername` with your GitHub username)

2. **Update `_config.yml`:**
   ```yaml
   url: https://yourusername.github.io
   baseurl: ""  # Leave empty for username.github.io repos
   ```

3. **Initialize git** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

4. **Add remote and push:**
   ```bash
   git remote add origin https://github.com/yourusername/yourusername.github.io.git
   git branch -M main
   git push -u origin main
   ```

### Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** → **Pages**
3. Under **Source**, select:
   - Branch: `main`
   - Folder: `/ (root)`
4. Click **Save**

Your site will be live at `https://yourusername.github.io` in a few minutes!

### Automatic Deployment

The repository includes a GitHub Actions workflow that automatically builds and deploys your site when you push to the `main` branch.

### Publishing New Posts

Simply create a new post, commit, and push:

```bash
# Create your post
vim _posts/2025-11-27-my-new-post.md

# Commit and push
git add _posts/2025-11-27-my-new-post.md
git commit -m "Add new post about diffusion models"
git push
```

GitHub Pages will automatically rebuild and deploy your site!

## Customizing Your Site

### Update Personal Information

Edit `_config.yml`:

```yaml
first_name: Your_First_Name
last_name: Your_Last_Name
description: >
  Your bio and description
blog_name: Your Blog Name
blog_description: Your blog description
```

### Update About Page

Edit `_pages/about.md` to customize your homepage.

### Add Profile Picture

Place your photo in `assets/img/` and reference it in `_pages/about.md`.

### Social Links

Update social media links in `_config.yml` under the social section (look for twitter, github, linkedin, etc.).

## Project Structure

```
.
├── _config.yml           # Site configuration
├── _posts/              # Blog posts (YYYY-MM-DD-title.md)
├── _pages/              # Static pages (about, publications, etc.)
├── _projects/           # Project showcase items
├── _news/               # News/announcements
├── assets/              # Images, PDFs, CSS, JS
│   ├── img/            # Images
│   ├── pdf/            # PDF files
│   └── json/           # JSON data (e.g., resume)
└── _bibliography/       # BibTeX files for publications
```

## Troubleshooting

### Port Already in Use

If port 8080 (Docker) or 4000 (native) is already in use:

**Docker:**
```bash
docker-compose down
docker-compose up
```

**Native:**
```bash
bundle exec jekyll serve --port 4001
```

### Changes Not Showing

1. **Hard refresh** your browser: `Cmd+Shift+R` (Mac) or `Ctrl+Shift+R` (Windows/Linux)
2. **Restart the server** if you changed `_config.yml` (config changes require restart)
3. **Clear browser cache**

### Math Not Rendering

1. Ensure `enable_math: true` in `_config.yml`
2. Use `$$ ... $$` for inline math (with spaces around)
3. Check browser console for JavaScript errors
4. Verify MathJax is loading (check page source for MathJax script)

### Build Errors

**Check your YAML front matter** - ensure all posts have:
- Valid YAML syntax (proper indentation)
- Required fields: `layout`, `title`, `date`
- Correct date format

**Check for special characters** in post content that might conflict with YAML or Markdown

## Example Post

Check out `_posts/2025-11-27-introduction-to-diffusion-models.md` for a complete example with:
- Proper front matter
- LaTeX equations (inline and display)
- Numbered equations with references
- Good structure and formatting

## Resources

- [al-folio Documentation](https://github.com/alshedivat/al-folio)
- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [MathJax Documentation](https://docs.mathjax.org/)
- [Markdown Guide](https://www.markdownguide.org/)
- [LaTeX Math Symbols](https://oeis.org/wiki/List_of_LaTeX_mathematical_symbols)

## Getting Help

- **al-folio Issues**: https://github.com/alshedivat/al-folio/issues
- **Jekyll Issues**: https://github.com/jekyll/jekyll/issues
- **MathJax Issues**: https://github.com/mathjax/MathJax/issues

## Next Steps

1. ✅ Review the example diffusion models post
2. ✅ Customize `_config.yml` with your information
3. ✅ Edit `_pages/about.md` to create your homepage
4. ✅ Add your profile picture to `assets/img/`
5. ✅ Create your first blog post
6. ✅ Set up GitHub Pages for deployment
7. ✅ Share your site!

Happy blogging! 🚀
