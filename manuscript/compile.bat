@echo off
echo ============================================
echo  Compiling RFOE manuscript for JSS
echo ============================================
echo.
echo Step 1/4: First pdflatex pass...
pdflatex -interaction=nonstopmode manuscript_jss.tex
echo.
echo Step 2/4: BibTeX pass (resolves references)...
bibtex manuscript_jss
echo.
echo Step 3/4: Second pdflatex pass...
pdflatex -interaction=nonstopmode manuscript_jss.tex
echo.
echo Step 4/4: Third pdflatex pass (final references)...
pdflatex -interaction=nonstopmode manuscript_jss.tex
echo.
echo ============================================
echo  Done! Output: manuscript_jss.pdf
echo ============================================
pause
