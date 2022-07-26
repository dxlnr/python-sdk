with open("build/html/index.html", "r") as file:
    filedata = file.read()
    filedata = filedata.replace(
        '<span class="caption-text"><a href="#content">content</a></span>',
        '<span class="caption-text"><a href="#content">Contents</a></span>',
    )
    filedata = filedata.replace(
        '<router-link to="#" class="home-link">',
        '<router-link to="https://modalic.ai" class="home-link">',
    )

with open("build/html/index.html", "w") as file:
    file.write(filedata)
