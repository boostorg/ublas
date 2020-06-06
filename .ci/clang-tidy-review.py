#!/usr/bin/env python3

# clang-tidy review
# Copyright (c) 2020 Peter Hill, Ashar Khan
# SPDX-License-Identifier: MIT
# See LICENSE for more information

import argparse
import itertools
import fnmatch
import json
import os
from operator import itemgetter
import pprint
import subprocess
import requests
import unidiff
from github import Github


def make_file_line_lookup(diff):
    """Get a lookup table for each file in diff, to convert between source
    line number to line number in the diff

    """
    lookup = {}
    for file in diff:
        filename = file.target_file[2:]
        lookup[filename] = {}
        pos = 1
        for hunk in file:
            for line in hunk:
                if not line.is_removed:
                    lookup[filename][line.target_line_no] = pos
                pos += 1
            pos += 1
    return lookup


def make_review(contents, lookup):
    """Construct a Github PR review given some warnings and a lookup table
    """
    root = os.getcwd()
    comments = []
    for num, line in enumerate(contents):
        if "warning" in line:
            full_path, source_line, _, warning = line.split(":", maxsplit=3)
            rel_path = os.path.relpath(full_path, root)
            body = ""
            for line2 in contents[num + 1 :]:
                if "warning" in line2:
                    break
                body += "\n" + line2.replace(full_path, rel_path)

            comment_body = f"""{warning.strip().replace("'", "`")}

```cpp
{body.strip()}
```
"""
            comments.append(
                {
                    "path": rel_path,
                    "body": comment_body,
                    "position": lookup[rel_path][int(source_line)],
                }
            )

    review = {
        "body": "clang-tidy made some suggestions",
        "event": "COMMENT",
        "comments": comments,
    }
    return review


def get_pr_diff(repo, pr_number, token):
    """Download the PR diff

    """

    headers = {
        "Accept": "application/vnd.github.v3.diff",
        "Authorization": f"token {token}",
    }
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"

    pr_diff_response = requests.get(url, headers=headers)
    pr_diff_response.raise_for_status()

    return unidiff.PatchSet(pr_diff_response.text)


def get_line_ranges(diff):
    """Return the line ranges of added lines in diff, suitable for the
    line-filter argument of clang-tidy

    """
    lines_by_file = {}
    for filename in diff:
        added_lines = []
        for hunk in filename:
            for line in hunk:
                if line.is_added:
                    added_lines.append(line.target_line_no)

        for _, group in itertools.groupby(
            enumerate(added_lines), lambda ix: ix[0] - ix[1]
        ):
            groups = list(map(itemgetter(1), group))
            lines_by_file.setdefault(filename.target_file[2:], []).append(
                [groups[0], groups[-1]]
            )

    line_filter_json = []
    for name, lines in lines_by_file.items():
        line_filter_json.append(str({"name": name, "lines": lines}))
    return json.dumps(line_filter_json, separators=(",", ":"))


def get_clang_tidy_warnings(
    line_filter, clang_tidy_binary, files
):
    """Get the clang-tidy warnings
    """

    command = f"{clang_tidy_binary} -line-filter={line_filter} {files} -- -Iinclude -std=c++17"
    print(f"Running:\n\t{command}")

    try:
        child = subprocess.run(command, capture_output=True, shell=True, check=True,)
    except subprocess.CalledProcessError as e:
        print(
            f"\n\nclang-tidy failed with return code {e.returncode} and error:\n{e.stderr.decode('utf-8')} and stdout:\n{e.stdout.decode('utf-8')}"
        )
        raise

    output = child.stdout.decode("utf-8", "ignore")

    return output.splitlines()


def post_lgtm_comment(pull_request):
    """Post a "LGTM" comment if everything's clean, making sure not to spam

    """

    BODY = 'clang-tidy review says "All clean, LGTM! :+1:"'

    comments = pull_request.get_issue_comments()

    for comment in comments:
        if comment.body == BODY:
            print("Already posted, no need to update")
            return

    pull_request.create_issue_comment(BODY)


def post_review(pull_request, review):
    """Post the review on the pull_request, making sure not to spam

    """

    comments = pull_request.get_review_comments()

    for comment in comments:
        review["comments"] = list(
            filter(
                lambda review_comment: not (
                    review_comment["path"] == comment.path
                    and review_comment["position"] == comment.position
                    and review_comment["body"] == comment.body
                ),
                review["comments"],
            )
        )

    print(f"::set-output name=total_comments::{len(review['comments'])}")

    if review["comments"] == []:
        print("Everything already posted!")
        return

    print("\nNew comments to post:")
    pprint.pprint(review)

    pull_request.create_review(**review)


def main(
    repo,
    pr_number,
    clang_tidy_binary,
    token,
    include,
    exclude,
):

    diff = get_pr_diff(repo, pr_number, token)
    print(f"\nDiff from GitHub PR:\n{diff}\n")

    line_ranges = get_line_ranges(diff)
    print(f"Line filter for clang-tidy:\n{line_ranges}\n")

    changed_files = [filename.target_file[2:] for filename in diff]
    files = []
    for pattern in include:
        files.extend(fnmatch.filter(changed_files, pattern))
    if exclude is None:
        exclude = []
    for pattern in exclude:
        files = [f for f in files if not fnmatch.fnmatch(f, pattern)]

    if files == []:
        print("No files to check!")
        return

    clang_tidy_warnings = get_clang_tidy_warnings(
        line_ranges, clang_tidy_binary, " ".join(files)
    )

    lookup = make_file_line_lookup(diff)
    review = make_review(clang_tidy_warnings, lookup)

    github = Github(token)
    repo = github.get_repo(f"{repo}")
    pull_request = repo.get_pull(pr_number)

    if review["comments"] == []:
        post_lgtm_comment(pull_request)
        return

    print("Posting a review:")
    pprint.pprint(review)
    post_review(pull_request, review)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a review from clang-tidy warnings"
    )
    parser.add_argument("--repo", help="Repo name in form 'owner/repo'")
    parser.add_argument("--pr", help="PR number", type=int)
    parser.add_argument(
        "--clang_tidy_binary", help="clang-tidy binary", default="clang-tidy-9"
    )
    parser.add_argument(
        "--include",
        help="Comma-separated list of files or patterns to include",
        type=str,
        nargs="?",
        default="*.[ch],*.[ch]xx,*.[ch]pp,*.[ch]++,*.cc,*.hh",
    )
    parser.add_argument(
        "--exclude",
        help="Comma-separated list of files or patterns to exclude",
        nargs="?",
        default="",
    )
    parser.add_argument("--token", help="github auth token")

    args = parser.parse_args()

    exclude = args.exclude.split(",") if args.exclude is not None else None

    main(
        repo=args.repo,
        pr_number=args.pr,
        clang_tidy_binary=args.clang_tidy_binary,
        token=args.token,
        include=args.include.split(","),
        exclude=exclude,
    )
