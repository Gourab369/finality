//! The sandbox launcher as a standalone binary, so tests can exercise it without the `finality` binary
//! (which runs the same code for its `__sandbox` argument).

fn main() {
    std::process::exit(finality_sandbox::launcher_main(
        std::env::args_os().skip(1).collect(),
    ));
}
