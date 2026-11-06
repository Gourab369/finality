//! Rule-table tests for the shell command analyzer (DESIGN §12.3).
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;

use finality_safety::{
    Category, CommandAnalyzer, CommandShape, EnvSnapshot, Location, PathGuard, ShellKind,
};

use Category::*;
use Location::{Inside as I, Outside as O, Unverifiable as U};

struct Fixture {
    _tmp: tempfile::TempDir,
    guard: PathGuard,
    env: EnvSnapshot,
}

fn fixture() -> Fixture {
    let tmp = tempfile::tempdir().unwrap();
    let root: PathBuf = dunce::canonicalize(tmp.path()).unwrap();
    for d in ["work/src", "work/sub", "other"] {
        std::fs::create_dir_all(root.join(d)).unwrap();
    }
    let guard = PathGuard::new(&root.join("work")).unwrap();
    let s = |p: PathBuf| p.display().to_string();
    let env = EnvSnapshot::from_pairs([
        ("HOME".to_owned(), s(root.clone())),
        ("USERPROFILE".to_owned(), s(root.clone())),
        ("TEMP".to_owned(), s(root.join("other"))),
        ("OUTSIDE".to_owned(), s(root.join("other"))),
        ("INSIDE".to_owned(), s(root.join("work/src"))),
    ]);
    Fixture {
        _tmp: tmp,
        guard,
        env,
    }
}

type Case = (&'static str, Location, Option<Category>);

fn run(shell: ShellKind, cases: &[Case]) {
    let f = fixture();
    let analyzer = CommandAnalyzer::new(&f.guard, &f.env);
    let mut failures = Vec::new();
    for (cmd, want, category) in cases {
        let a = analyzer.analyze(shell, cmd);
        let category_ok = category.is_none_or(|c| a.has(c));
        if a.location != *want || !category_ok {
            failures.push(format!(
                "{shell:?} `{cmd}`: want {want:?} {category:?}, got {:?} {:?}",
                a.location, a.findings
            ));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn posix_rules() {
    run(
        ShellKind::Posix,
        &[
            // directory change
            ("cd .. && ls", O, Some(DirectoryChange)),
            ("cd /tmp; rm -rf x", O, Some(DirectoryChange)),
            ("cd", O, Some(DirectoryChange)),
            ("cd ~ && ls", O, Some(DirectoryChange)),
            ("pushd ../other", O, Some(DirectoryChange)),
            ("cd -", U, Some(DirectoryChange)),
            ("cd $UNSET_DIR", U, Some(DirectoryChange)),
            ("cd src && cat ../../other/x", O, Some(OutsidePath)),
            ("cd src && cargo test", I, None),
            ("cd src; ls main.rs", I, None),
            // Every possible cwd is kept, so `..` after an inside cd is checked from the workdir too.
            ("cd src; ls ../src", O, Some(OutsidePath)),
            // dir-redirect flags
            ("git -C ../other status", O, Some(DirFlag)),
            ("make -C /tmp", O, Some(DirFlag)),
            ("tar -C .. -xf a.tar", O, Some(DirFlag)),
            ("npm --prefix ../other install", O, Some(DirFlag)),
            (
                "cargo build --manifest-path=../other/Cargo.toml",
                O,
                Some(DirFlag),
            ),
            ("node app.js --cwd /", O, Some(DirFlag)),
            ("git -C src status", I, None),
            ("make --directory=src all", I, None),
            // outside paths
            ("cat /etc/passwd", O, Some(OutsidePath)),
            ("rm -rf ../other", O, Some(OutsidePath)),
            ("cp src/a ../other/", O, Some(OutsidePath)),
            ("ls ~/", O, Some(OutsidePath)),
            ("vim ~/.bashrc", O, Some(OutsidePath)),
            ("grep -r x ../", O, Some(OutsidePath)),
            ("gcc -I/usr/include main.c", O, Some(OutsidePath)),
            ("/usr/bin/env ls", O, Some(OutsidePath)),
            ("cat src/main.rs", I, None),
            ("ls -la", I, None),
            ("rm -rf target", I, None),
            ("curl https://example.com/a/b", I, None),
            ("grep -rn 'foo/bar' src", I, None),
            ("./build.sh --release", I, None),
            // variable expansion
            ("cat $HOME/.ssh/id_rsa", O, Some(OutsidePath)),
            ("ls $OUTSIDE", O, Some(OutsidePath)),
            ("cat ${HOME}/x", O, Some(OutsidePath)),
            ("rm -rf $UNSET_VAR/x", U, Some(VariableExpansion)),
            ("cat ~root/.ssh/id_rsa", U, Some(VariableExpansion)),
            ("OUT=../other make", O, Some(OutsidePath)),
            ("ls $INSIDE", I, None),
            ("echo $INSIDE/x", I, None),
            // nested interpreters
            ("bash -c 'rm -rf /'", U, Some(NestedInterpreter)),
            ("sh -c ls", U, Some(NestedInterpreter)),
            ("eval ls", U, Some(NestedInterpreter)),
            ("python -c 'import os'", U, Some(NestedInterpreter)),
            ("node -e 'x'", U, Some(NestedInterpreter)),
            ("find . -exec rm {} \\;", U, Some(NestedInterpreter)),
            ("ls | xargs rm", U, Some(NestedInterpreter)),
            ("perl -e 1", U, Some(NestedInterpreter)),
            ("$CMD foo", U, Some(NestedInterpreter)),
            ("env -i ls", U, Some(NestedInterpreter)),
            ("python script.py", I, None),
            ("node app.js", I, None),
            ("bash ./build.sh", I, None),
            ("find src -name '*.rs'", I, None),
            ("nohup cargo build", I, None),
            ("timeout 10 cargo test", I, None),
            // substitution
            ("echo $(cat /etc/passwd)", O, Some(OutsidePath)),
            ("ls `pwd`", U, Some(Substitution)),
            ("diff <(sort a) b", U, Some(Substitution)),
            ("rm $(find . -name x)", U, Some(Substitution)),
            ("echo $((1+2))", U, Some(Substitution)),
            ("echo $(cd ..)", O, Some(DirectoryChange)),
            // redirection
            ("echo x > /tmp/out", O, Some(Redirection)),
            ("ls >> ../log", O, Some(Redirection)),
            ("make 2> ~/err", O, Some(Redirection)),
            ("cat < /etc/hosts", O, Some(Redirection)),
            ("echo hi > $UNSET/x", U, Some(Redirection)),
            ("make > build.log 2>&1", I, None),
            ("ls 2>/dev/null", I, None),
            // link creation
            ("ln -s ../other link", O, Some(LinkCreation)),
            ("ln -s /etc x", O, Some(LinkCreation)),
            ("ln target link", U, Some(LinkCreation)),
            ("ln -sf src/a b", U, Some(LinkCreation)),
            ("mount /dev/sda1 x", O, Some(LinkCreation)),
            // privilege / remote
            ("sudo rm x", O, Some(Privilege)),
            ("su -c ls", O, Some(Privilege)),
            ("doas ls", O, Some(Privilege)),
            ("ssh host ls", O, Some(Remote)),
            ("scp a host:b", O, Some(Remote)),
            ("rsync -a src/ host:/x", O, Some(Remote)),
            ("rsync -a src/ backup/", I, None),
            // unparseable
            ("echo 'x", U, Some(Unparseable)),
            ("cat <<EOF\nx\nEOF", U, Some(Unparseable)),
            ("echo $(ls", U, Some(Unparseable)),
            ("cat >", U, Some(Unparseable)),
            ("echo \"x", U, Some(Unparseable)),
        ],
    );
}

#[test]
fn powershell_rules() {
    run(
        ShellKind::PowerShell,
        &[
            // directory change
            ("Set-Location ..", O, Some(DirectoryChange)),
            ("cd /", O, Some(DirectoryChange)),
            ("sl ../other", O, Some(DirectoryChange)),
            ("Push-Location ~", O, Some(DirectoryChange)),
            (
                "Set-Location -Path ../other; Get-ChildItem",
                O,
                Some(DirectoryChange),
            ),
            ("Set-Location $env:UNSET", U, Some(DirectoryChange)),
            ("Set-Location src; Get-ChildItem", I, None),
            // dir-redirect flags
            ("git -C ../other log", O, Some(DirFlag)),
            (
                "Start-Process app.exe -WorkingDirectory ../other",
                O,
                Some(DirFlag),
            ),
            ("npm --prefix ../other test", O, Some(DirFlag)),
            ("make -C ..", O, Some(DirFlag)),
            (
                "cargo test --manifest-path ../other/Cargo.toml",
                O,
                Some(DirFlag),
            ),
            ("git -C src log", I, None),
            // outside paths
            ("Get-Content ../other/x", O, Some(OutsidePath)),
            ("Remove-Item -Recurse -Force ../other", O, Some(OutsidePath)),
            ("Copy-Item src/a /tmp/b", O, Some(OutsidePath)),
            ("Get-ChildItem ~/Documents", O, Some(OutsidePath)),
            ("Get-ChildItem Env:", O, Some(OutsidePath)),
            ("Get-Content -Path:../x", O, Some(OutsidePath)),
            (
                "Set-ItemProperty HKLM:/Software/x -Name a -Value 1",
                O,
                Some(OutsidePath),
            ),
            ("Get-Content src/main.rs", I, None),
            ("Remove-Item target -Recurse", I, None),
            (
                "Invoke-WebRequest https://example.com/x -OutFile out.html",
                I,
                None,
            ),
            // variable expansion
            ("Get-Content $env:USERPROFILE/x", O, Some(OutsidePath)),
            ("Get-ChildItem $env:OUTSIDE", O, Some(OutsidePath)),
            ("Get-Content $HOME/x", O, Some(OutsidePath)),
            ("Get-Content \"${env:TEMP}/x\"", O, Some(OutsidePath)),
            ("Remove-Item $env:UNSET/x", U, Some(VariableExpansion)),
            ("Remove-Item $target", U, Some(VariableExpansion)),
            ("Get-ChildItem $env:INSIDE", I, None),
            ("Get-ChildItem | ForEach-Object { $_.Name }", I, None),
            (
                "Get-ChildItem src | Where-Object { $_.Length -gt 0 }",
                I,
                None,
            ),
            // nested interpreters
            ("Invoke-Expression 'rm x'", U, Some(NestedInterpreter)),
            ("iex $code", U, Some(NestedInterpreter)),
            ("pwsh -c 'Get-ChildItem'", U, Some(NestedInterpreter)),
            (
                "powershell -EncodedCommand AAAA",
                U,
                Some(NestedInterpreter),
            ),
            ("cmd /c dir", U, Some(NestedInterpreter)),
            ("bash -c ls", U, Some(NestedInterpreter)),
            ("Start-Job { Get-Date }", U, Some(NestedInterpreter)),
            (
                "Invoke-Command -ScriptBlock { ls }",
                U,
                Some(NestedInterpreter),
            ),
            ("pwsh -File ./build.ps1", I, None),
            ("& ./build.ps1", I, None),
            ("python script.py", I, None),
            // substitution
            ("Write-Output $(Get-Content ../x)", O, Some(OutsidePath)),
            (
                "Remove-Item \"$(Get-Location)/../x\"",
                U,
                Some(Substitution),
            ),
            (
                "Get-ChildItem | ForEach-Object { Remove-Item ../x }",
                O,
                Some(OutsidePath),
            ),
            ("Get-Item @(Get-Content /etc/hosts)", O, Some(OutsidePath)),
            ("Write-Output $(cd ..)", O, Some(DirectoryChange)),
            ("Write-Output $(Get-Date)", I, None),
            // redirection
            ("Get-Process > ../out.txt", O, Some(Redirection)),
            ("Get-Date >> /tmp/x", O, Some(Redirection)),
            ("dir 2> ~/err.txt", O, Some(Redirection)),
            ("echo x *> ../all.txt", O, Some(Redirection)),
            ("Get-Date > $env:UNSET/x", U, Some(Redirection)),
            ("Get-Date > out.txt", I, None),
            ("dir 2>&1 > $null", I, None),
            // link creation
            (
                "New-Item -ItemType SymbolicLink -Path x -Target ../other",
                O,
                Some(LinkCreation),
            ),
            ("ni -ItemType Junction x -Value src", U, Some(LinkCreation)),
            (
                "New-Item -ItemType HardLink -Name h -Value a",
                U,
                Some(LinkCreation),
            ),
            ("mklink x y", U, Some(LinkCreation)),
            ("subst X: src", O, Some(LinkCreation)),
            ("New-Item -ItemType Directory build", I, None),
            // privilege / remote
            ("Start-Process pwsh -Verb RunAs", O, Some(Privilege)),
            ("Enter-PSSession server", O, Some(Remote)),
            ("New-PSSession -ComputerName x", O, Some(Remote)),
            ("ssh host", O, Some(Remote)),
            ("runas /user:admin cmd", O, Some(Privilege)),
            // unparseable
            ("Get-Item 'x", U, Some(Unparseable)),
            ("Get-Item \"x", U, Some(Unparseable)),
            ("Write-Output $(Get-Date", U, Some(Unparseable)),
            ("}", U, Some(Unparseable)),
            ("Get-Date >", U, Some(Unparseable)),
        ],
    );
}

#[test]
fn cmd_rules() {
    run(
        ShellKind::Cmd,
        &[
            // directory change
            ("cd .. & dir", O, Some(DirectoryChange)),
            ("cd /d ..", O, Some(DirectoryChange)),
            ("chdir ../other", O, Some(DirectoryChange)),
            ("pushd ..", O, Some(DirectoryChange)),
            ("cd %USERPROFILE%", O, Some(DirectoryChange)),
            ("cd src && dir", I, None),
            ("cd", I, None),
            // dir-redirect flags
            ("git -C .. status", O, Some(DirFlag)),
            ("start /d .. app.exe", O, Some(DirFlag)),
            ("npm --prefix ../other test", O, Some(DirFlag)),
            ("make -C ../other", O, Some(DirFlag)),
            (
                "cargo build --manifest-path ../other/Cargo.toml",
                O,
                Some(DirFlag),
            ),
            ("git -C src status", I, None),
            // outside paths
            ("type ../other/x", O, Some(OutsidePath)),
            ("del /q ../other/x", O, Some(OutsidePath)),
            ("copy src/a ../other", O, Some(OutsidePath)),
            ("dir /s ..", O, Some(OutsidePath)),
            ("more /etc/hosts", O, Some(OutsidePath)),
            ("type src/main.rs", I, None),
            ("dir /s src", I, None),
            ("del /q /s target", I, None),
            // variable expansion
            ("type %USERPROFILE%/x", O, Some(OutsidePath)),
            ("dir %OUTSIDE%", O, Some(OutsidePath)),
            ("type %TEMP%/x", O, Some(OutsidePath)),
            ("del %UNSET%/x", U, Some(VariableExpansion)),
            ("echo %~dp0", U, Some(Substitution)),
            ("dir %INSIDE%", I, None),
            ("echo 50% done", I, None),
            // nested interpreters
            ("cmd /c dir", U, Some(NestedInterpreter)),
            ("cmd /k echo", U, Some(NestedInterpreter)),
            ("powershell -c Get-Date", U, Some(NestedInterpreter)),
            ("pwsh -Command x", U, Some(NestedInterpreter)),
            ("bash -c ls", U, Some(NestedInterpreter)),
            (
                "for /f %i in ('dir') do echo %i",
                U,
                Some(NestedInterpreter),
            ),
            ("if exist x del y", U, Some(NestedInterpreter)),
            // redirection
            ("dir > ../out.txt", O, Some(Redirection)),
            ("echo x >> /tmp/x", O, Some(Redirection)),
            ("type a 2> ../err", O, Some(Redirection)),
            ("sort < ../in.txt", O, Some(Redirection)),
            ("dir > %UNSET%/x", U, Some(Redirection)),
            ("dir > out.txt 2>&1", I, None),
            ("dir > NUL", I, None),
            // link creation
            ("mklink link ../other", O, Some(LinkCreation)),
            ("mklink /J j ../other", O, Some(LinkCreation)),
            ("mklink /D d src", U, Some(LinkCreation)),
            ("subst X: src", O, Some(LinkCreation)),
            ("net use Z: //server/share", O, Some(LinkCreation)),
            // privilege / remote
            ("runas /user:admin cmd", O, Some(Privilege)),
            ("gsudo del x", O, Some(Privilege)),
            ("psexec //host cmd", O, Some(Remote)),
            ("ssh host", O, Some(Remote)),
            ("scp a host:b", O, Some(Remote)),
            // unparseable
            ("echo \"x", U, Some(Unparseable)),
            ("dir >", U, Some(Unparseable)),
        ],
    );
}

#[cfg(windows)]
#[test]
fn windows_path_forms() {
    run(
        ShellKind::PowerShell,
        &[
            ("Get-Content ..\\other\\x", O, Some(OutsidePath)),
            ("Get-Content C:\\Windows\\win.ini", O, Some(OutsidePath)),
            ("Remove-Item \\\\server\\share\\x", O, Some(OutsidePath)),
            ("Get-Content \\\\?\\C:\\x", O, Some(OutsidePath)),
            ("Get-Content src\\main.rs", I, None),
        ],
    );
    run(
        ShellKind::Cmd,
        &[
            ("type %USERPROFILE%\\x", O, Some(OutsidePath)),
            ("type C:x", O, Some(OutsidePath)),
            ("del ..\\x", O, Some(OutsidePath)),
            ("cd \\", O, Some(DirectoryChange)),
            ("type src\\a.rs", I, None),
        ],
    );
}

/// The examples named in the Phase 3 acceptance criteria.
#[test]
fn acceptance_examples_are_not_inside() {
    let f = fixture();
    let a = CommandAnalyzer::new(&f.guard, &f.env);
    for (shell, cmd) in [
        (ShellKind::Posix, "git -C ../x status"),
        (ShellKind::Posix, "cd .. ; ls"),
        (ShellKind::PowerShell, "pwsh -c \"Get-ChildItem\""),
        (ShellKind::Posix, "echo $(cat /etc/passwd)"),
        (ShellKind::Cmd, "type %USERPROFILE%\\x"),
    ] {
        assert_ne!(a.analyze(shell, cmd).location, I, "{shell:?} `{cmd}`");
    }
}

#[test]
fn grant_keys_cover_each_simple_command() {
    let f = fixture();
    let a = CommandAnalyzer::new(&f.guard, &f.env);
    let r = a.analyze(
        ShellKind::Posix,
        "cargo test --release && git status | wc -l",
    );
    assert_eq!(r.grant_keys, vec!["cargo test", "git status", "wc"]);
    let r = a.analyze(
        ShellKind::PowerShell,
        "Get-ChildItem src; npm.cmd run build",
    );
    assert_eq!(r.grant_keys, vec!["get-childitem src", "npm run"]);
}

#[test]
fn shapes_normalize_options_quotes_and_powershell_aliases() {
    let f = fixture();
    let a = CommandAnalyzer::new(&f.guard, &f.env);
    let names = |s: &CommandShape| {
        (
            s.name.clone(),
            s.short.iter().collect::<String>(),
            s.long.iter().cloned().collect::<Vec<_>>(),
            s.positional.clone(),
        )
    };
    let r = a.analyze(ShellKind::Posix, "rm -r'f' target && echo $(rm -f -r x)");
    let shapes: Vec<_> = r.shapes.iter().map(names).collect();
    assert_eq!(
        shapes[..2],
        [
            ("rm".into(), "fr".into(), vec![], vec!["target".into()]),
            ("rm".into(), "fr".into(), vec![], vec!["x".into()]),
        ],
        "substituted commands get their own shape"
    );
    assert_eq!(shapes[2].0, "echo");
    let r = a.analyze(
        ShellKind::PowerShell,
        "rm -r -fo .\\build; Remove-Item -Recurse x",
    );
    let shapes: Vec<_> = r.shapes.iter().map(names).collect();
    assert_eq!(
        shapes,
        [
            (
                "remove-item".into(),
                String::new(),
                vec!["-force".into(), "-recurse".into()],
                vec![".\\build".into()]
            ),
            (
                "remove-item".into(),
                String::new(),
                vec!["-recurse".into()],
                vec!["x".into()]
            ),
        ]
    );
    let r = a.analyze(ShellKind::Cmd, "rd /s /q build");
    assert_eq!(
        names(&r.shapes[0]),
        (
            "rd".into(),
            String::new(),
            vec!["/q".into(), "/s".into()],
            vec!["build".into()]
        )
    );
}
