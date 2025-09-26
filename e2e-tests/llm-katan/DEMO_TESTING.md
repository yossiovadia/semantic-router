# Testing the Interactive Demo

## For PR Reviewers

Since the live demo links won't work until this PR is merged, here are ways to see the animation:

### Option 1: Quick Local Test
```bash
# Check out the PR branch
git fetch origin pull/240/head:terminal-demo-test
git checkout terminal-demo-test

# Open the demo in your browser
open e2e-tests/llm-katan/terminal-demo.html
# or on Linux: xdg-open e2e-tests/llm-katan/terminal-demo.html
```

### Option 2: View Raw File
1. Go to the [terminal-demo.html file in this PR](https://github.com/vllm-project/semantic-router/pull/240/files#diff-terminal-demo.html)
2. Click "View file"
3. Copy the content to a local `.html` file
4. Open in browser

### Option 3: Static Preview
The README already includes a collapsible preview showing the terminal output - this works immediately in the PR!

## What You'll See
- 3-panel terminal layout
- Realistic typing animations
- Multi-instance setup demonstration
- Professional terminal styling
- Complete workflow from install to testing

The demo showcases llm-katan's key feature: running one tiny model as multiple different AI providers for testing.