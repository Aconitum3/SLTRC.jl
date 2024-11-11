using SLTRC
using Documenter

DocMeta.setdocmeta!(SLTRC, :DocTestSetup, :(using SLTRC); recursive=true)

makedocs(;
    modules=[SLTRC],
    authors="aconitum3 <aconitum@example.com> and contributors",
    sitename="SLTRC.jl",
    format=Documenter.HTML(;
        canonical="https://aconitum3.github.io/SLTRC.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/aconitum3/SLTRC.jl",
    devbranch="main",
)
