cross build --release --target x86_64-unknown-linux-gnu
cp target/x86_64-unknown-linux-gnu/release/duosplit target/x86_64-unknown-linux-gnu/release/duosplit-x86_64-unknown-linux-gnu
cross build --release --target aarch64-unknown-linux-gnu
cp target/aarch64-unknown-linux-gnu/release/duosplit target/aarch64-unknown-linux-gnu/release/duosplit-aarch64-unknown-linux-gnu
cross build --release --target x86_64-pc-windows-gnu
cp target/x86_64-pc-windows-gnu/release/duosplit.exe target/x86_64-pc-windows-gnu/release/duosplit-x86_64-pc-windows-gnu.exe