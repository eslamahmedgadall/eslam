import 'package:flutter/material.dart';
import 'package:splash_project/introduction_screen/presentetion/view/widgets/intro_screen_body.dart';

class IntroductionScreen extends StatelessWidget {
  const IntroductionScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IntroScreenBody(),
    );
  }
}
