#! usr/bin/env python3

# Automatically create review from clang-tidy report
# Copyright (c) 2020 Ashar Khan <ashar786khan.at.gmail.com>
# Licensed under Boost Software License 1.0

import argparse
from github import Github
import unidiff
import os
import requests


def fetch_diff(repository, pr, token):
    """
    Fetch the diff of PR from repository using Github Authentication token 
    """
    headers = {
        "Accept": "application/vnd.github.v3.diff",
        "Authorization": f"token {token}",
    }
    url = f"https://api.github.com/repos/{repository}/pulls/{pr}"
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    return unidiff.PatchSet(response.text)


def parse_report_file(content):
    """
    Parses clang-tidy stdout content to a list of tuple(relpath, line, comment), this tuple is called reports throughout the docs.
        relpath: The relative path of source that generated clang-tidy warning or error
        line: Line number in the above soure to which the error or warning belongs
        comment: The actual clang-tidy error or warning for above line and source
    """
    root_path = os.getcwd()
    parsed_report = []
    n = len(content)
    i = 0
    while (i < n):
        if "error" in content[i] or "warning" in content[i]:
            abs_file_path, line_number, _, severity, message = content[i].split(
                ":", maxsplit=4)
            relative_path = os.path.relpath(abs_file_path, root_path)

            k = 1
            body = ""
            while i+k < n and not ("error" in content[i+k] or "warning" in content[i+k]):
                body += "\n" + content[i + k].replace(abs_file_path, relative_path)
                k += 1
            i += k

            message = message.strip().replace("'", "`")
            body = body.strip()

            comment_text = f"""{message}
```cpp
{body}
```
Reported as{severity} by clang-tidy.
"""
            parsed_report.append((relative_path, line_number, comment_text))
    return parsed_report


def file_filter_report(reports, diff):
    """
    Filter the reports so that it only contains the reports for files in diff that have changed or been added in the PR
    """
    changed_files = []
    for file in diff:
        changed_files.append(file.target_file[2:])

    filtered_report = []
    for report in reports:
        rel_path, line_number, comment_text = report
        if rel_path in changed_files:
            filtered_report.append(report)

    return filtered_report


def line_filter_report(reports, diff):
    """
    Filter the reports so that it only contains the reports for lines in the diff that have been added.
    """
    files_in_report = [i[0] for i in reports]
    filtered_report = []

    lookup = {}
    for file in diff:
        if file.target_file[2:] in files_in_report:
            for hunk in file:
                for line in hunk.target_lines():
                    if line.is_added:
                        if file.target_file[2:] not in lookup:
                            lookup[file.target_file[2:]] = list()
                        lookup[file.target_file[2:]].append(
                            int(line.target_line_no))

    for report in reports:
        fileName, number, _ = report
        number = int(number)
        if number in lookup[fileName]:
            filtered_report.append(report)

    return filtered_report


def line_to_position(reports, diff):
    """
    The actual source file line number in report is different from what we need to send to github as review payload.
    This function converts the clang-tidy generated error or warning line numbers to position in diff, 
    so that review can be made for position relevant to the PR.
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
    new_reports = []
    for report in reports:
        x, y, z = report
        new_reports.append((x, lookup[x][int(y)], z))

    return new_reports


def comment_lgtm(pr_handle):
    """
    Posts a LGTM (Looks good to me!) comment in the PR, if PR did not produced new clang-tidy warnings or errors.
    """
    lgtm = 'This Pull request Passed all of clang-tidy tests. :+1:'
    comments = pr_handle.get_issue_comments()

    for comment in comments:
        if comment.body == lgtm:
            print("Already posted LGTM!!")
            return

    pr_handle.create_issue_comment(lgtm)


def post_review(reports, pr_handle):
    """
    Posts the reports as review to the PR
    """
    comments = pr_handle.get_review_comments()
    for comment in comments:
        target = (comment.path, comment.position, comment.body)
        if target in reports:
            reports.remove((comment.path, comment.position, comment.body))

    if len(reports) == 0:
        comment_lgtm(pr_handle)
    else:
        comments = []
        for report in reports:
            comments.append(
                {"path": report[0], "position": report[1], "body": report[2]})

        review = {
            "body": "Reports by clang tidy",
            "event": "COMMENT",
            "comments": comments,
        }

        pr_handle.create_review(**review)
        print("Review comments are on its way!")


def main(pathToClangReport, repository, pr, token):
    """ 
    Psudo-Main function, that calls all above functions in order.
    """
    f = open(pathToClangReport)
    lines = f.read().splitlines()
    f.close()

    reports = parse_report_file(lines)
    diff = fetch_diff(repository, pr, token)
    # Filter the report, So that only changed files are commented
    reports = file_filter_report(reports, diff)
    # Fileter the reports, So that only changed lines are commented
    reports = line_filter_report(reports, diff)

    github = Github(token)
    repo = github.get_repo(f"{repository}")
    pr_handle = repo.get_pull(pr)

    if len(reports) == 0:
        comment_lgtm(pr_handle)
    else:
        post_review(line_to_position(reports, diff), pr_handle)


if __name__ == "__main__":
    """
    Parses command line arguments for the script and calls main function with the script.
    """
    parser = argparse.ArgumentParser(
        description="Create review from clang tidy file")
    parser.add_argument(
        "--path", help="Path to clang-tidy report file. This file can be"
        "generated by redirecting stdout of clang-tidy to a file while running the check", required=True)
    parser.add_argument(
        "--repository", help="Pass repositor on Github with owner/repo format", required=True)
    parser.add_argument(
        "--pr", help="The repository's PR number to perform review", type=int, required=True)
    parser.add_argument(
        "--token", help="Github authentication token", required=True)

    args = parser.parse_args()

    main(args.path, args.repository, args.pr, args.token)
