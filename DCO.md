# ✅ Fixing DCO Check Failures

The **Developer Certificate of Origin (DCO)** check ensures all commits are signed off.
 If your PR fails the DCO check, here’s how to fix it.

---

## 🖥️ Option 1: Fix via GitHub Web Editor
 ⚠️ Works only if your PR has 1 commit.

1. Go to your **Pull Request** → **Commits** tab.
2. Click the **⋯ menu** → **Edit commit message**.
3. Add this line at the end of the commit message:

   ```text
   Signed-off-by: Your Name <your.email@example.com>
   ```
4. Save changes → GitHub will create a new commit with sign-off.
5. Re-run the DCO check.

## 📦 Option 2: Fix via GitHub Desktop

1. Open your branch in GitHub Desktop.
2. Go to Repository → Repository Settings → Commit Behavior.
3. Check ✅ Always sign-off commits.
4. Amend the last commit:
      - Right-click the commit → Amend Commit.
      - Save again with sign-off enabled.
5. Push with force (if required):
    ```
    git push --force-with-lease
    ```

## 💻 Option 3: Fix via CLI (Multiple Commits)

1. Enable sign-off in your config:
   ```
    git config --global user.name "Your Name"
    git config --global user.email "your.email@example.com"
   ```
2. Re-sign commits interactively:
   ```
    git rebase -i HEAD~N
   ```
   Replace N with the number of commits to fix.
   Mark commits as edit, then run:
   ```
    git commit --amend --signoff
    git rebase --continue
   ```
3. Push with:
   ```
    git push --force-with-lease
   ```

## 🔀 If Your Branch Is Messy After Syncing with main

- Simplest fix: squash all commits into a single new signed commit (via Desktop or CLI).
- Alternatively, ask a maintainer to Squash and Merge with a sign-off on merge.

## ✨ Pro Tips
This ensures you’ll never fail DCO again.

- Use the -s flag when committing from CLI:
   ```
     git commit -s -m "Your commit message"
   ```
- Turn on Always sign-off commits in your client (GitHub Desktop or Git CLI).
    1. GitHub Desktop
       Turn on Always sign-off commits in your client.

    2. Git CLI
       You can always sign-off commits automatically using a commit template (NOTE: This will only work if you use enter your commit message interactively with `git commit`, and will _not_ work with `git commit -m "<message>"`):
       1. Create ~/.git-commit-template.txt with:
        ```
          Signed-off-by: Your Name <your.email@example.com>
         ```
    2. Tell Git to use it:
       ```
        git config --global commit.template ~/.git-commit-template.txt
       ```
