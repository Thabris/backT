# Development Guidelines for BackT

## Git Commit Policy

⚠️ **IMPORTANT: DO NOT COMMIT WITHOUT EXPLICIT REQUEST**

### Commit Rules
- **NEVER commit automatically** after creating or modifying files
- **ALWAYS wait for explicit approval** before running `git add` or `git commit`
- **TEST AND VALIDATE** all changes before considering commits
- Only commit when specifically asked: "please commit this" or "push to git"

### Development Workflow
1. Create/modify files as requested
2. **TEST the changes locally first**
3. Validate functionality works as expected
4. Wait for explicit commit instruction
5. Only then add, commit, and push to repository

### Testing Requirements
- Run examples to ensure they work
- Check imports and dependencies
- Validate output format and content
- Ensure no breaking changes to existing functionality

### Rationale
- Prevents pushing untested/broken code
- Allows for iterative development and refinement
- Maintains clean git history with working commits
- Enables proper testing before integration

## Example Workflow

```bash
# ❌ WRONG - Don't do this automatically
python new_example.py  # Create file
git add .
git commit -m "Add new example"

# ✅ CORRECT - Do this instead
python new_example.py  # Create file
# Test the file first
python examples/new_example.py  # Validate it works
# Wait for explicit approval
# Only commit when asked
```

## Current Status
This policy is now in effect. All future development should follow these guidelines.